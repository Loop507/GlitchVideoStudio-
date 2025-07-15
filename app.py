import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import os
import random
from PIL import Image
from pathlib import Path

# Configurazione pagina
st.set_page_config(page_title="🎥 Glitch Video Studio", page_icon="🎥", layout="wide")

def generate_placeholder_image(width=640, height=480):
    """Genera un'immagine RGB casuale (simula un effetto glitch iniziale)"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def apply_glitch_with_ffmpeg(input_path, output_path, duration=5, fps=30):
    """
    Usa FFmpeg per applicare effetti glitch e generare un video
    """
    cmd = [
        'ffmpeg',
        '-y',  # Sovrascrivi file esistente
        '-loop', '1', '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-t', str(duration),
        '-r', str(fps),
        '-vf', f"fps={fps},"
               "geq='rand(0)*255':128:128,"
               "scale=2*iw:-1,crop=iw/2:ih,"
               "hflip,"
               "eq=contrast=1.5:brightness=0.1,"
               "format=gray,"  # Simula rumore VHS
               "vignette=FX=0.2,"
               "gltransition=duration=0.5:source=fadeblack.mp4",
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error("Errore durante la generazione del video con FFmpeg")
        st.code(e.stderr.decode())
        return False

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
        st.image(cv2.cvtColor(apply_pixel_shuffle(img_array.copy(), intensity=5), cv2.COLOR_BGR2RGB), caption="Pixel Shuffle", use_column_width=True)

    st.markdown("---")

    if st.button("🎥 Genera Video Glitch", type="primary", use_container_width=True):
        with st.spinner("Sto generando il video..."):

            # Salva l'immagine temporaneamente
            temp_dir = tempfile.TemporaryDirectory()
            input_img_path = Path(temp_dir.name) / "input.jpg"
            output_video_path = Path(temp_dir.name) / "output.mp4"

            cv2.imwrite(str(input_img_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

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
