import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import random
from pathlib import Path
from PIL import Image
import math
import json

st.set_page_config(page_title="🎥 Glitch Video Studio", page_icon="🎥", layout="wide")

# === CACHING ===
@st.cache_data(show_spinner=False)
def cached_resize(img, fmt):
    return resize_to_format(img, fmt)

# === EFFETTI GLITCH (in HSV/LAB/Time-based support) ===
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

def apply_rgb_shift(frame, i, total_frames, max_shift=5):
    h, w = frame.shape[:2]
    dynamic_shift = int(max_shift * math.sin(2 * math.pi * i / total_frames))
    shift_x = random.randint(-dynamic_shift, dynamic_shift)
    shift_y = random.randint(-dynamic_shift, dynamic_shift)
    b, g, r = cv2.split(frame)
    def shift(c, dx, dy):
        return cv2.warpAffine(c, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([shift(b, -shift_x, -shift_y), shift(g, 0, 0), shift(r, shift_x, shift_y)])

def apply_color_inversion(frame):
    return cv2.bitwise_not(frame)

def apply_analog_noise(frame, amount=0.1):
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    noise = np.random.randn(*lab.shape).astype(np.float32) * amount * 10
    lab += noise
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_scanlines(frame):
    for y in range(0, frame.shape[0], 2):
        frame[y:y+1, :] = frame[y:y+1, :] // 2
    return frame

def apply_hue_shift(frame, i, total_frames, shift_base=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + int(shift_base * math.sin(i / total_frames * 2 * math.pi))) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_datamosh(frame, prev_frame):
    blend = cv2.addWeighted(frame, 0.5, prev_frame, 0.5, 0)
    return blend

# === MOTION ===
def apply_base_motion(frame, frame_idx, total_frames, motion_type, motion_intensity, motion_speed):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    t = frame_idx / total_frames * motion_speed
    angle = scale = tx = ty = 0
    if motion_type in ["rotate", "mix"]:
        angle = math.sin(t * 2 * math.pi) * 5 * motion_intensity
    if motion_type in ["zoom", "mix"]:
        scale = 1 + 0.02 * math.cos(t * 2 * math.pi) * motion_intensity
    else:
        scale = 1
    if motion_type in ["translate", "mix"]:
        tx = 5 * math.sin(t * 2 * math.pi * 0.5) * motion_intensity
        ty = 5 * math.cos(t * 2 * math.pi * 0.5) * motion_intensity
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# === FORMAT ===
def resize_to_format(img_np, fmt):
    h, w = img_np.shape[:2]
    target_ratio = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}.get(fmt, 16 / 9)
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        img_np = img_np[:, offset:offset+new_w]
    else:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        img_np = img_np[offset:offset+new_h, :]
    return img_np

# === PRESET HANDLING ===
def save_preset(preset):
    with open("glitch_preset.json", "w") as f:
        json.dump(preset, f)

def load_preset():
    try:
        with open("glitch_preset.json", "r") as f:
            return json.load(f)
    except:
        return None

# === FRAME GENERATION ===
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    prev_frame = img_np.copy()
    progress_bar = st.progress(0)
    for i in range(n_frames):
        frame = apply_base_motion(img_np.copy(), i, n_frames, settings['motion_type'], settings['motion_intensity'], settings['motion_speed'])
        if settings['pixel_shuffle']: frame = apply_pixel_shuffle(frame, int(10 * settings['intensity']))
        if settings['rgb_shift']: frame = apply_rgb_shift(frame, i, n_frames, 5)
        if settings['invert']: frame = apply_color_inversion(frame)
        if settings['noise']: frame = apply_analog_noise(frame, 0.1 * settings['intensity'])
        if settings['scanlines']: frame = apply_scanlines(frame)
        if settings['hue_shift']: frame = apply_hue_shift(frame, i, n_frames, 30)
        if settings['datamosh'] and i % 7 == 0: frame = apply_datamosh(frame, prev_frame)
        prev_frame = frame.copy()
        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        progress_bar.progress((i + 1) / n_frames)

# === VIDEO ===
def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)

# === MAIN ===
def main():
    st.title("🎥 Glitch Video Studio")
    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    video_format = st.sidebar.selectbox("Formato video", ["16:9", "9:16", "1:1"])
    duration = st.sidebar.slider("Durata (s)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Salva preset"):
        save_preset(st.session_state.get("settings", {}))
    if st.sidebar.button("📂 Carica preset"):
        preset = load_preset()
        if preset:
            st.session_state.settings = preset
            st.experimental_rerun()

    st.sidebar.markdown("---")
    intensity = st.sidebar.slider("Intensità effetti", 0.1, 2.0, 1.0, 0.1)
    speed = st.sidebar.slider("Velocità glitch", 0.1, 2.0, 1.0, 0.1)
    color = st.sidebar.slider("Colore", 0.1, 2.0, 1.0, 0.1)
    motion_type = st.sidebar.selectbox("Tipo movimento", ["mix", "rotate", "zoom", "translate"])
    motion_intensity = st.sidebar.slider("Intensità movimento", 0.0, 3.0, 1.0, 0.1)
    motion_speed = st.sidebar.slider("Velocità movimento", 0.1, 3.0, 1.0, 0.1)
    st.sidebar.markdown("---")
    pixel_shuffle = st.sidebar.checkbox("Pixel Shuffle", value=True)
    rgb_shift = st.sidebar.checkbox("RGB Shift", value=True)
    invert = st.sidebar.checkbox("Inversione Colori")
    noise = st.sidebar.checkbox("Noise Analogico")
    scanlines = st.sidebar.checkbox("Scanlines CRT")
    hue_shift = st.sidebar.checkbox("Hue Psichedelico")
    datamosh = st.sidebar.checkbox("Datamosh Light")

    settings = {
        'intensity': intensity,
        'speed': speed,
        'color': color,
        'motion_type': motion_type,
        'motion_intensity': motion_intensity,
        'motion_speed': motion_speed,
        'pixel_shuffle': pixel_shuffle,
        'rgb_shift': rgb_shift,
        'invert': invert,
        'noise': noise,
        'scanlines': scanlines,
        'hue_shift': hue_shift,
        'datamosh': datamosh
    }
    st.session_state.settings = settings

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        img_np = cached_resize(np.array(img), video_format)
        st.image(img_np, caption="Anteprima immagine", use_container_width=True)

        if st.button("🎬 Genera video glitch"):
            with st.spinner("Creazione in corso..."):
                temp_dir = tempfile.TemporaryDirectory()
                output_dir = Path(temp_dir.name)
                n_frames = duration * fps
                generate_glitch_frames(img_np, n_frames, output_dir, settings)
                video_path = output_dir / "glitch_output.mp4"
                try:
                    generate_video_from_frames(video_path, fps, output_dir)
                    st.success("✅ Video creato con successo!")
                    with open(video_path, "rb") as vf:
                        st.video(vf.read())
                        st.download_button("⬇️ Scarica Video", vf, file_name="glitch_output.mp4")
                except subprocess.CalledProcessError:
                    st.error("Errore durante la generazione del video.")
                finally:
                    temp_dir.cleanup()
    else:
        st.warning("📷 Carica un'immagine per iniziare.")

if __name__ == "__main__":
    main()
