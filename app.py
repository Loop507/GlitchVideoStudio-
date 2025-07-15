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

# --- EFFETTI GLITCH ---
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

def apply_rgb_shift(frame, max_shift=10):
    h, w = frame.shape[:2]
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    b, g, r = cv2.split(frame)
    def shift(c, dx, dy):
        return cv2.warpAffine(c, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b, shift(g, 0, shift_y), shift(r, shift_x, 0)])

def apply_color_inversion(frame, intensity=1.0):
    inverted = cv2.bitwise_not(frame)
    return cv2.addWeighted(frame, 1 - intensity, inverted, intensity, 0)

def apply_analog_noise(frame, amount=0.1):
    noise = np.random.randn(*frame.shape) * 255 * amount
    noisy = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_scanlines(frame):
    for y in range(0, frame.shape[0], 2):
        frame[y:y+1, :] = frame[y:y+1, :] // 2
    return frame

def apply_posterization(frame, levels=4):
    div = 256 // levels
    return (frame // div * div).astype(np.uint8)

# --- GENERA FRAME ---
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    intensity_map = {'Soft': 3, 'Medium': 7, 'Hard': 15}
    glitch_intensity = intensity_map.get(settings.get('glitch_strength', 'Medium'), 7)

    for i in range(n_frames):
        frame = img_np.copy()
        apply_random = settings.get('random_mode', False)

        if apply_random:
            effects = [
                apply_pixel_shuffle, apply_rgb_shift, apply_color_inversion, apply_analog_noise,
                apply_scanlines, apply_posterization
            ]
            random.shuffle(effects)
            for effect in effects[:random.randint(3, 6)]:
                if effect == apply_color_inversion:
                    frame = effect(frame, intensity=settings.get('invert_intensity', 0.5))
                elif effect == apply_pixel_shuffle:
                    frame = effect(frame, intensity=glitch_intensity)
                elif effect == apply_analog_noise:
                    frame = effect(frame, amount=0.05 * glitch_intensity)
                else:
                    frame = effect(frame)
        else:
            if settings['pixel_shuffle']:
                frame = apply_pixel_shuffle(frame, intensity=glitch_intensity)
            if settings['rgb_shift']:
                frame = apply_rgb_shift(frame)
            if settings['invert']:
                frame = apply_color_inversion(frame, intensity=settings.get('invert_intensity', 0.5))
            if settings['noise']:
                frame = apply_analog_noise(frame, amount=0.05 * glitch_intensity)
            if settings['scanlines']:
                frame = apply_scanlines(frame)
            if settings['posterize']:
                frame = apply_posterization(frame)

        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        progress_bar.progress((i + 1) / n_frames)

# --- GENERA VIDEO ---
def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

# --- RIDIMENSIONA E PADDA ---
def resize_with_padding(img, target_ratio):
    h, w = img.shape[:2]
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-2:
        return img

    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)

    resized_img = cv2.resize(img, (w, h))
    top = (new_h - h) // 2
    bottom = new_h - h - top
    left = (new_w - w) // 2
    right = new_w - w - left
    color = [0, 0, 0]

    padded = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

# --- MAIN ---
def main():
    st.title("🎥 Glitch Video Studio")
    st.sidebar.header("⚙️ Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    resolution = st.sidebar.selectbox("Risoluzione", options=["1:1", "9:16", "16:9"], index=0)
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Effetti Glitch")

    settings = {
        'pixel_shuffle': st.sidebar.checkbox("Pixel Shuffle", value=True),
        'rgb_shift': st.sidebar.checkbox("RGB Shift", value=True),
        'invert': st.sidebar.checkbox("Color Inversion", value=False),
        'invert_intensity': st.sidebar.slider("Intensità Color Inversion", 0.0, 1.0, 0.0, step=0.1),
        'noise': st.sidebar.checkbox("Analog Noise + Grain", value=False),
        'scanlines': st.sidebar.checkbox("Scanlines CRT", value=False),
        'posterize': st.sidebar.checkbox("Posterize + Contrast", value=False),
        'random_mode': st.sidebar.checkbox("Random Mode (Glitch Mix)", value=False),
        'glitch_strength': st.sidebar.select_slider("Forza Glitch", options=["Soft", "Medium", "Hard"], value="Medium"),
    }

    if not uploaded_img:
        st.warning("Carica un'immagine per iniziare")
        return

    img = Image.open(uploaded_img).convert('RGB')
    img_np = np.array(img)

    # Calcola rapporto di risoluzione scelto
    ratios = {"1:1": 1.0, "9:16": 9/16, "16:9": 16/9}
    target_ratio = ratios.get(resolution, 1.0)

    img_np_resized = resize_with_padding(img_np, target_ratio)

    st.image(img_np_resized, caption="Immagine caricata ridimensionata", use_container_width=True)

    if st.sidebar.button("🎬 Genera Video"):
        with st.spinner("Generazione video glitch in corso..."):
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = Path(temp_dir.name)
            n_frames = duration * fps

            generate_glitch_frames(img_np_resized, n_frames, output_dir, settings)

            video_path = output_dir / "glitch_video.mp4"
            try:
                generate_video_from_frames(video_path, fps, output_dir)
                st.success("✅ Video generato!")
                with open(video_path, "rb") as vf:
                    st.video(vf.read())
                    st.download_button("Scarica Video", data=vf, file_name="glitch_video.mp4")
            except subprocess.CalledProcessError:
                st.error("Errore durante la generazione del video.")
            finally:
                temp_dir.cleanup()

if __name__ == "__main__":
    main()
