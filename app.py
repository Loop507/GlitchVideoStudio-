import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import os
import random
from pathlib import Path
from PIL import Image

# Configurazione pagina
st.set_page_config(page_title="🎥 Glitch Video Studio", page_icon="🎥", layout="wide")


# === FUNZIONI UTILI ===

def apply_pixel_shuffle(frame, intensity=5):
    """Effetto Pixel Shuffle - Sposta blocchi di pixel in modo casuale"""
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


def generate_placeholder_image(width=640, height=480):
    """Genera un'immagine RGB casuale come placeholder"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def apply_glitch_with_ffmpeg(input_path, output_path, duration=5, fps=30):
    """
    Usa FFmpeg per applicare effetti glitch e generare un video.
    Richiede ffmpeg installato nell'ambiente.
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-loop', '1', '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-t', str(duration),
        '-r', str(fps),
        '-vf', f"fps={fps},"
               "scale=2*iw:-1,crop=iw/2:ih,"
               "hflip,"
               "eq=contrast=1.5:brightness=0.1,"
               "noise=alls=20:allf=t,"
               "format=gray,"
               "vignette",
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        st.error("❌ Errore durante la generazione del video con FFmpeg")
        if e.stderr:
            st.code(e.stderr.decode())
        else:
            st.code("Errore sconosciuto: nessun output da FFmpeg.")
        return False


# === MAIN APP ===

def main():
    st.title("🎥 Glitch Video Studio")
    st.markdown("*by Loop507*")
    st.markdown("---")

    st.sidebar.header("⚙️ Impostazioni")
    uploaded_file = st.sidebar.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])

    duration = st.sidebar.slider("Durata video (secondi)", 1, 30, 5)
    fps = st.sidebar.slider("Frame per secondo", 10, 60, 30)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Immagine originale")
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(img)
            st.image(img, caption="Immagine caricata", use_column_width=True)
        else:
            img_array = generate_placeholder_image()
            st.warning("Nessuna immagine caricata: uso un'immagine casuale.")
            st.image(img_array, caption="Immagine di default", use_column_width=True)

    with col2:
        st.subheader("🔍 Anteprima Effetti")
        preview_img = apply_pixel_shuffle(img_array.copy(), intensity=5)
        st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="Pixel Shuffle", use_column_width=True)

    st.markdown("---")

    if st.button("🎥 Genera Video Glitch", type="primary", use_container_width=True):
        with st.spinner("Sto generando il video..."):

            # Crea una directory temporanea
            temp_dir = tempfile.TemporaryDirectory()
            input_img_path = Path(temp_dir.name) / "input.jpg"
            output_video_path = Path(temp_dir.name) / "output.mp4"

            # Salva l'immagine temporaneamente
            cv2.imwrite(str(input_img_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            # Genera il video glitch
            success = apply_glitch_with_ffmpeg(input_img_path, output_video_path, duration=duration, fps=fps)

            if success and output_video_path.exists():
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()

                st.success("✅ Video generato con successo!")
                st.download_button(
                    label="⬇️ Scarica Video Glitch",
                    data=video_bytes,
                    file_name=f"glitch_video_{duration}s.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("❌ Impossibile generare il video.")

            temp_dir.cleanup()

    else:
        st.info("👆 Carica un'immagine o usa quella di default per iniziare!")


if __name__ == "__main__":
    main()
