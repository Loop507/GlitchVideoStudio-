import streamlit as st
import cv2
import numpy as np
import random
import os
import tempfile
from PIL import Image

# Configura la pagina
st.set_page_config(
    page_title="Glitch Video Studio",
    page_icon="🎥",
    layout="wide"
)

def apply_pixel_shuffle(frame, intensity=5):
    height, width = frame.shape[:2]
    block_size = max(1, intensity)
    blocks = []
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            end_y = min(y + block_size, height)
            end_x = min(x + block_size, width)
            block = frame[y:end_y, x:end_x]
            
            if block.shape[0] > 0 and block.shape[1] > 0:
                blocks.append((x, y, block))
    
    if not blocks:
        return frame
    
    random.shuffle(blocks)
    new_frame = np.zeros_like(frame)
    
    for i, (orig_x, orig_y, block) in enumerate(blocks):
        idx = random.randint(0, len(blocks) - 1)
        target_x, target_y, _ = blocks[idx]
        
        block_h, block_w = block.shape[:2]
        end_y = min(target_y + block_h, height)
        end_x = min(target_x + block_w, width)
        
        if end_y - target_y < block_h or end_x - target_x < block_w:
            block = block[:end_y - target_y, :end_x - target_x]
        
        new_frame[target_y:end_y, target_x:end_x] = block
    
    return new_frame

def apply_color_shift(frame, intensity=20):
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return frame
    
    b, g, r = cv2.split(frame)
    b = np.roll(b, shift=random.randint(-intensity, intensity), axis=1)
    r = np.roll(r, shift=random.randint(-intensity, intensity), axis=1)
    return cv2.merge([b, g, r])

def apply_scanlines(frame, intensity=1):
    height, width = frame.shape[:2]
    step = max(2, random.randint(2, 5))
    
    for y in range(0, height, step):
        if y < height:
            darkness = random.randint(20, 50)
            frame[y:min(y+1, height), :] = np.clip(
                frame[y:min(y+1, height), :].astype(np.int16) - darkness, 
                0, 255
            ).astype(np.uint8)
    
    return frame

def apply_vhs_noise(frame, intensity=5):
    height, width = frame.shape[:2]
    noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    _, mask = cv2.threshold(noise, 230, 255, cv2.THRESH_BINARY)
    
    # Crea un rumore bianco con lo stesso numero di canali del frame
    if len(frame.shape) == 3:  # Immagine a colori (3 canali)
        white_noise = np.zeros_like(frame)
        white_noise[mask == 255] = [255, 255, 255]
    else:  # Scala di grigi (1 canale)
        white_noise = np.zeros_like(frame)
        white_noise[mask == 255] = 255
    
    return cv2.addWeighted(frame, 0.8, white_noise, 0.2, 0)

def create_glitch_video_from_image(image_array, duration=5, fps=30):
    height, width = image_array.shape[:2]
    
    # Crea un file temporaneo per il video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec più compatibile
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError("Impossibile creare il file video")
    
    num_frames = fps * duration
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_frames):
        frame = image_array.copy()
        
        # Applica gli effetti glitch
        if i % 3 == 0:
            frame = apply_pixel_shuffle(frame, intensity=random.randint(3, 8))
        if i % 4 == 0:
            frame = apply_color_shift(frame, intensity=random.randint(10, 30))
        if i % 6 == 0:
            frame = apply_scanlines(frame)
        if i % 8 == 0:
            frame = apply_vhs_noise(frame)
        
        out.write(frame)
        
        # Aggiorna la progress bar
        progress = (i + 1) / num_frames
        progress_bar.progress(progress)
        status_text.text(f"Generando frame {i+1}/{num_frames}")
    
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path

def main():
    st.title("🎥 Glitch Video Studio")
    st.markdown("*by Loop507*")
    st.markdown("---")
    
    # Sidebar per i controlli
    st.sidebar.header("⚙️ Impostazioni")
    
    # Upload dell'immagine
    uploaded_file = st.sidebar.file_uploader(
        "Carica un'immagine",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Converti l'immagine in RGB e poi in BGR per OpenCV
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Sidebar controls
        duration = st.sidebar.slider("Durata video (secondi)", 1, 30, 5)
        fps = st.sidebar.slider("Frame per secondo", 10, 60, 30)
        
        # Layout a colonne
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Immagine originale")
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.subheader("🎬 Anteprima effetti")
            
            # Mostra anteprime degli effetti
            preview_frame = image_array.copy()
            
            effect_tabs = st.tabs(["Pixel Shuffle", "Color Shift", "Scanlines", "VHS Noise"])
            
            with effect_tabs[0]:
                preview_shuffle = apply_pixel_shuffle(preview_frame.copy(), intensity=5)
                preview_shuffle_rgb = cv2.cvtColor(preview_shuffle, cv2.COLOR_BGR2RGB)
                st.image(preview_shuffle_rgb, use_column_width=True)
            
            with effect_tabs[1]:
                preview_color = apply_color_shift(preview_frame.copy(), intensity=20)
                preview_color_rgb = cv2.cvtColor(preview_color, cv2.COLOR_BGR2RGB)
                st.image(preview_color_rgb, use_column_width=True)
            
            with effect_tabs[2]:
                preview_scan = apply_scanlines(preview_frame.copy())
                preview_scan_rgb = cv2.cvtColor(preview_scan, cv2.COLOR_BGR2RGB)
                st.image(preview_scan_rgb, use_column_width=True)
            
            with effect_tabs[3]:
                preview_vhs = apply_vhs_noise(preview_frame.copy())
                preview_vhs_rgb = cv2.cvtColor(preview_vhs, cv2.COLOR_BGR2RGB)
                st.image(preview_vhs_rgb, use_column_width=True)
        
        # Pulsante per generare il video
        st.markdown("---")
        
        if st.button("🎥 Genera Video Glitch", type="primary", use_container_width=True):
            with st.spinner("Generando video glitch..."):
                try:
                    video_path = create_glitch_video_from_image(
                        image_array, 
                        duration=duration, 
                        fps=fps
                    )
                    
                    st.success("✅ Video generato con successo!")
                    
                    # Leggi il file video e permetti il download
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    st.download_button(
                        label="⬇️ Scarica Video Glitch",
                        data=video_bytes,
                        file_name=f"glitch_video_{duration}s.mp4",
                        mime="video/mp4"
                    )
                    
                    # Pulisci il file temporaneo
                    os.unlink(video_path)
                    
                except Exception as e:
                    st.error(f"❌ Errore durante la generazione: {str(e)}")
    
    else:
        st.info("👆 Carica un'immagine dalla sidebar per iniziare!")
        
        # Mostra esempio
        st.markdown("---")
        st.subheader("🎯 Come funziona")
        st.markdown("""
        1. **Carica un'immagine** dalla sidebar
        2. **Regola le impostazioni** (durata e FPS)
        3. **Visualizza l'anteprima** degli effetti
        4. **Genera il video** con gli effetti glitch
        5. **Scarica il risultato** sul tuo dispositivo
        """)

if __name__ == "__main__":
    main()
