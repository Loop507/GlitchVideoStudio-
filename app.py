import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import random
from pathlib import Path
from PIL import Image
import math

st.set_page_config(page_title="🎥 Glitch Video Studio", page_icon="🎥", layout="wide")

def apply_pixel_shuffle(frame, intensity=5):
    h, w = frame.shape[:2]
    block = max(1, int(min(h, w) / intensity))
    new_frame = frame.copy()
    for _ in range(intensity * 2):
        x = random.randint(0, w - block)
        y = random.randint(0, h - block)
        dx = random.randint(-block, block)
        dy = random.randint(-block, block)
        src = frame[y:y+block, x:x+block].copy()
        tx = np.clip(x + dx, 0, w - block)
        ty = np.clip(y + dy, 0, h - block)
        new_frame[ty:ty+block, tx:tx+block] = src
    return new_frame

def apply_dynamic_pixel_shuffle(frame, frame_index, max_intensity=20):
    intensity = int((math.sin(frame_index * 0.1) + 1) / 2 * max_intensity) + 1
    return apply_pixel_shuffle(frame, intensity=intensity)

def generate_glitch_frames(img_np, n_frames, output_dir):
    progress_bar = st.progress(0)
    for i in range(n_frames):
        frame = apply_dynamic_pixel_shuffle(img_np.copy(), i)
        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        progress_bar.progress((i + 1) / n_frames)

def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    st.title("🎥 Glitch Video Studio")
    st.sidebar.header("⚙️ Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])

    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)
    generate_btn = st.sidebar.button("🎬 Genera Video")

    if not uploaded_img:
        st.warning("⚠️ Carica un'immagine per iniziare")
        return

    img = Image.open(uploaded_img).convert('RGB')
    img_np = np.array(img)
    st.image(img_np, caption="Immagine caricata", use_container_width=True)

    if generate_btn:
        with st.spinner("🎞 Generazione in corso..."):
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = Path(temp_dir.name)

            n_frames = duration * fps

            generate_glitch_frames(img_np, n_frames, output_dir)
            video_path = output_dir / "glitch_video.mp4"

            try:
                generate_video_from_frames(video_path, fps, output_dir)
                st.success("✅ Video generato!")
                video_file = open(video_path, "rb")
                st.video(video_file.read())
                st.download_button("⬇️ Scarica Video", data=video_file, file_name="glitch_video.mp4")
            except subprocess.CalledProcessError:
                st.error("❌ Errore durante la generazione del video.")
            finally:
                temp_dir.cleanup()

if __name__ == "__main__":
    main()
