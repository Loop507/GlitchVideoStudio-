import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import os
import random
from pathlib import Path
from PIL import Image

# Config pagina
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


def generate_glitch_frames(img_np, n_frames, output_dir):
    progress_bar = st.progress(0)
    for i in range(n_frames):
        intensity = random.randint(5, 15)
        frame = apply_pixel_shuffle(img_np.copy(), intensity=intensity)
        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        progress_bar.progress((i + 1) / n_frames)


def get_audio_duration(audio_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(output)
    except:
        return None


def generate_video_from_frames(output_path, frame_rate, temp_dir, audio_path=None):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p'
    ]
    if audio_path:
        cmd += ['-i', str(audio_path), '-shortest', '-c:a', 'aac']
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def main():
    st.title("🎥 Glitch Video Studio")
    st.sidebar.header("⚙️ Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    uploaded_audio = st.sidebar.file_uploader("Carica audio (opzionale)", type=["mp3", "wav"])

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
            audio_path = None
            audio_duration = None

            if uploaded_audio:
                audio_path = output_dir / "audio.mp3"
                with open(audio_path, 'wb') as f:
                    f.write(uploaded_audio.read())
                audio_duration = get_audio_duration(audio_path)

            video_duration = int(audio_duration) if audio_duration else duration
            n_frames = video_duration * fps

            generate_glitch_frames(img_np, n_frames, output_dir)
            video_path = output_dir / "glitch_video.mp4"

            try:
                generate_video_from_frames(video_path, fps, output_dir, audio_path)
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
