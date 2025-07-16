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

# === EFFETTI GLITCH ===
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

def apply_rgb_shift(frame, max_shift=5):
    h, w = frame.shape[:2]
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    b, g, r = cv2.split(frame)
    def shift(c, dx, dy):
        return cv2.warpAffine(c, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([shift(b, -shift_x, -shift_y), shift(g, 0, 0), shift(r, shift_x, shift_y)])

def apply_color_inversion(frame):
    return cv2.bitwise_not(frame)

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

def apply_jpeg_artifacts(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    _, encimg = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_row_column_shift(frame):
    h, w = frame.shape[:2]
    frame = frame.copy()
    for _ in range(5):
        row = random.randint(0, h - 2)
        shift = random.randint(-10, 10)
        frame[row] = np.roll(frame[row], shift, axis=0)
    return frame

def apply_wave_distortion(frame):
    h, w = frame.shape[:2]
    distorted = np.zeros_like(frame)
    for y in range(h):
        offset = int(10.0 * math.sin(2 * math.pi * y / 64 + random.uniform(-0.5, 0.5)))
        distorted[y] = np.roll(frame[y], offset, axis=0)
    return distorted

def apply_pixel_stretch(frame):
    h, w = frame.shape[:2]
    stretched = frame.copy()
    for _ in range(5):
        y = random.randint(0, h - 1)
        stretched[y] = stretched[y, random.randint(0, w - 1)]
    return stretched

def apply_edge_overlay(frame):
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(frame, 0.8, edge_colored, 0.2, 0)

def apply_hue_shift(frame, shift=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_glitch_grid(frame):
    h, w = frame.shape[:2]
    grid = frame.copy()
    step = 20
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), (0, 255, 255), 1)
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), (0, 255, 255), 1)
    return cv2.addWeighted(frame, 0.9, grid, 0.1, 0)

def apply_vhs_effect(frame):
    frame = apply_scanlines(frame)
    frame = apply_wave_distortion(frame)
    frame = apply_analog_noise(frame, 0.03)
    return frame

def apply_base_motion(frame, frame_idx, total_frames, motion_intensity, motion_speed):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    t = frame_idx / total_frames * motion_speed
    angle = math.sin(t * 2 * math.pi) * 5 * motion_intensity
    scale = 1 + 0.02 * math.cos(t * 2 * math.pi) * motion_intensity
    tx = 5 * math.sin(t * 2 * math.pi * 0.5) * motion_intensity
    ty = 5 * math.cos(t * 2 * math.pi * 0.5) * motion_intensity
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    moved = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return moved

# RIDIMENSIONAMENTO IMMAGINE IN BASE AL FORMATO

def resize_to_format(img_np, fmt):
    h, w = img_np.shape[:2]
    if fmt == "16:9":
        target_ratio = 16 / 9
    elif fmt == "9:16":
        target_ratio = 9 / 16
    else:
        target_ratio = 1.0
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

# === GENERA FRAME ===
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    for i in range(n_frames):
        base_frame = apply_base_motion(img_np.copy(), i, n_frames, settings['motion_intensity'], settings['motion_speed'])
        frame = base_frame.copy()

        if settings['pixel_shuffle']: frame = apply_pixel_shuffle(frame, settings['pixel_shuffle_int'])
        if settings['rgb_shift']: frame = apply_rgb_shift(frame, settings['rgb_shift_int'])
        if settings['invert']: frame = apply_color_inversion(frame)
        if settings['noise']: frame = apply_analog_noise(frame, settings['noise_int'])
        if settings['scanlines']: frame = apply_scanlines(frame)
        if settings['posterize']: frame = apply_posterization(frame, settings['posterize_lvl'])
        if settings['hue_shift']: frame = apply_hue_shift(frame, settings['hue_shift_val'])
        if settings['glitch_grid']: frame = apply_glitch_grid(frame)
        if settings['jpeg']: frame = apply_jpeg_artifacts(frame)
        if settings['rowcol']: frame = apply_row_column_shift(frame)
        if settings['wave']: frame = apply_wave_distortion(frame)
        if settings['stretch']: frame = apply_pixel_stretch(frame)
        if settings['edge']: frame = apply_edge_overlay(frame)
        if settings['vhs']: frame = apply_vhs_effect(frame)

        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        progress_bar.progress((i + 1) / n_frames)

# === GENERA VIDEO ===
def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)

# === MAIN APP ===
def main():
    st.title(":camera: Glitch Video Studio")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)
    video_format = st.sidebar.selectbox("Formato video", ["16:9", "9:16", "1:1"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader(":control_knobs: Controlli Globali")
    global_intensity = st.sidebar.slider("Intensità Globale", 0.1, 2.0, 1.0, 0.1)
    global_speed = st.sidebar.slider("Velocità Glitch", 0.1, 2.0, 1.0, 0.1)
    global_color = st.sidebar.slider("Saturazione Colore", 0.0, 2.0, 1.0, 0.1)
    motion_intensity = st.sidebar.slider("Intensità Movimento", 0.0, 3.0, 1.0, 0.1)
    motion_speed = st.sidebar.slider("Velocità Movimento", 0.1, 3.0, 1.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader(":game_die: Effetti")

    settings = {
        'pixel_shuffle': st.sidebar.checkbox("Pixel Shuffle", value=True),
        'pixel_shuffle_int': int(10 * global_intensity),
        'rgb_shift': st.sidebar.checkbox("RGB Shift", value=True),
        'rgb_shift_int': int(5 * global_intensity),
        'invert': st.sidebar.checkbox("Color Inversion", value=False),
        'noise': st.sidebar.checkbox("Analog Noise + Grain", value=False),
        'noise_int': 0.1 * global_intensity,
        'scanlines': st.sidebar.checkbox("Scanlines CRT", value=False),
        'posterize': st.sidebar.checkbox("Posterize + Contrast", value=False),
        'posterize_lvl': max(2, int(6 / global_intensity)),
        'hue_shift': st.sidebar.checkbox("Hue Shift Psichedelico", value=False),
        'hue_shift_val': int(30 * global_intensity),
        'glitch_grid': st.sidebar.checkbox("Glitch Grid Overlay", value=False),
        'jpeg': st.sidebar.checkbox("JPEG Artifacts", value=False),
        'rowcol': st.sidebar.checkbox("Row/Column Shift", value=False),
        'wave': st.sidebar.checkbox("Wave Distortion", value=False),
        'stretch': st.sidebar.checkbox("Pixel Stretch", value=False),
        'edge': st.sidebar.checkbox("Edge Overlay", value=False),
        'vhs': st.sidebar.checkbox("VHS Effect", value=False),
        'motion_intensity': motion_intensity,
        'motion_speed': motion_speed
    }

    if uploaded_img:
        img = Image.open(uploaded_img).convert('RGB')
        img_np = resize_to_format(np.array(img), video_format)
        st.image(img_np, caption="Immagine caricata", use_container_width=True)

        if st.button(":clapper: Avvia generazione"):
            with st.spinner("Generazione glitch video..."):
                temp_dir = tempfile.TemporaryDirectory()
                output_dir = Path(temp_dir.name)
                n_frames = duration * fps

                generate_glitch_frames(img_np, n_frames, output_dir, settings)

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
    else:
        st.warning("Carica un'immagine per iniziare")

if __name__ == "__main__":
    main()
