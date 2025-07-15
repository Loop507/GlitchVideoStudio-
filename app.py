import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import random
from pathlib import Path
from PIL import Image
import math

st.set_page_config(page_title="🎥 Glitch Video Studio - Extra Effects", page_icon="🎥", layout="wide")

# === FUNZIONI UTILI ===

def resize_and_pad(img, target_ratio):
    h, w = img.shape[:2]
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.01:
        return img
    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (new_h - h) // 2
    bottom = new_h - h - top
    left = (new_w - w) // 2
    right = new_w - w - left
    padded = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return padded

def apply_3d_motion(frame, angle_deg, scale, tx, ty):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    M[:,2] += [tx, ty]
    moved = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return moved

# --- EFFETTI BASE ---

def apply_pixel_shuffle(frame, intensity=5):
    h, w = frame.shape[:2]
    block = max(1, int(min(h, w) / max(5, intensity)))
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

def apply_rgb_shift(frame, intensity=5):
    h, w = frame.shape[:2]
    max_shift = max(1, intensity)
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    b, g, r = cv2.split(frame)
    def shift(c, dx, dy):
        return cv2.warpAffine(c, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b, shift(g, 0, shift_y), shift(r, shift_x, 0)])

def apply_color_inversion(frame, intensity=1):
    inv = cv2.bitwise_not(frame)
    return cv2.addWeighted(frame, 1 - intensity, inv, intensity, 0)

def apply_analog_noise(frame, intensity=0.1):
    noise = np.random.randn(*frame.shape) * 255 * intensity
    noisy = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_scanlines(frame, intensity=0.5):
    h = frame.shape[0]
    step = max(1, int(2 / intensity)) if intensity > 0 else 1000
    for y in range(0, h, step):
        frame[y:y+1, :] = (frame[y:y+1, :] * (1 - intensity)).astype(np.uint8)
    return frame

def apply_posterization(frame, levels=4):
    div = max(1, 256 // levels)
    return (frame // div * div).astype(np.uint8)

def apply_pixel_stretch(frame, intensity=5):
    h, w = frame.shape[:2]
    stretched = frame.copy()
    for _ in range(intensity):
        y = random.randint(0, h - 1)
        col = random.randint(0, w - 1)
        stretched[y] = stretched[y, col]
    return stretched

def apply_row_column_shift(frame, intensity=5):
    h, w = frame.shape[:2]
    frame = frame.copy()
    for _ in range(intensity):
        row = random.randint(0, h - 2)
        shift = random.randint(-10, 10)
        frame[row] = np.roll(frame[row], shift, axis=0)
    return frame

def apply_edge_overlay(frame, intensity=0.2):
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(frame, 1 - intensity, edge_colored, intensity, 0)

def apply_hue_shift(frame, intensity=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + intensity) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# === EFFETTI EXTRA RICHIESTI ===

def apply_band_noise(frame, intensity=0.3):
    """Disturbo a bande colorate orizzontali frastagliate"""
    h, w = frame.shape[:2]
    band_height = max(2, int(h * 0.02))
    frame = frame.copy()
    for y in range(0, h, band_height * 2):
        if random.random() > 1 - intensity:
            noise = np.random.randint(0, 255, (band_height, w, 3), dtype=np.uint8)
            frame[y:y+band_height] = cv2.addWeighted(frame[y:y+band_height], 0.5, noise, 0.5, 0)
    return frame

def apply_broken_lines(frame, intensity=0.3):
    """Linee verticali e orizzontali spezzate b/n con tocchi di colore"""
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    step = max(3, int(20 * (1 - intensity)))
    for y in range(0, h, step):
        if random.random() < intensity:
            x_break = random.randint(0, w-10)
            frame[y, x_break:x_break+10] = 255 - frame[y, x_break:x_break+10]
    for x in range(0, w, step):
        if random.random() < intensity:
            y_break = random.randint(0, h-10)
            frame[y_break:y_break+10, x] = 255 - frame[y_break:y_break+10, x]
    return frame

def apply_digital_grid(frame, intensity=0.3):
    """Linee e forme geometriche in scala di grigi + tocchi di colore"""
    h, w = frame.shape[:2]
    grid = np.zeros_like(frame)
    step = max(10, int(50 * (1 - intensity)))
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), (int(255*intensity), int(255*intensity), int(255*intensity)), 1)
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), (int(255*intensity), int(255*intensity*0.7), int(255*intensity*0.7)), 1)
    return cv2.addWeighted(frame, 1 - intensity, grid, intensity, 0)

# === GENERA FRAME ===
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)

    for i in range(n_frames):
        frame = img_np.copy()

        # Movimento 3D organico della foto principale
        angle = 2 * math.sin(i / n_frames * 2 * math.pi) * settings['motion_angle']
        scale = 1 + 0.02 * math.sin(i / n_frames * 4 * math.pi) * settings['motion_zoom']
        tx = int(5 * math.sin(i / n_frames * 3 * math.pi) * settings['motion_tx'])
        ty = int(5 * math.sin(i / n_frames * 5 * math.pi) * settings['motion_ty'])
        frame = apply_3d_motion(frame, angle, scale, tx, ty)

        # Effetti glitch base con intensità
        if settings['pixel_shuffle']:
            frame = apply_pixel_shuffle(frame, intensity=settings['pixel_shuffle_intensity'])
        if settings['rgb_shift']:
            frame = apply_rgb_shift(frame, intensity=settings['rgb_shift_intensity'])
        if settings['invert']:
            frame = apply_color_inversion(frame, intensity=settings['invert_intensity'])
        if settings['noise']:
            frame = apply_analog_noise(frame, intensity=settings['noise_intensity'])
        if settings['scanlines']:
            frame = apply_scanlines(frame, intensity=settings['scanlines_intensity'])
        if settings['posterize']:
            frame = apply_posterization(frame, levels=settings['posterize_levels'])
        if settings['pixel_stretch']:
            frame = apply_pixel_stretch(frame, intensity=settings['pixel_stretch_intensity'])
        if settings['row_column_shift']:
            frame = apply_row_column_shift(frame, intensity=settings['row_column_shift_intensity'])
        if settings['edge_overlay']:
            frame = apply_edge_overlay(frame, intensity=settings['edge_overlay_intensity'])
        if settings['hue_shift']:
            frame = apply_hue_shift(frame, intensity=settings['hue_shift_intensity'])

        # Effetti extra
        if settings['band_noise']:
            frame = apply_band_noise(frame, intensity=settings['band_noise_intensity'])
        if settings['broken_lines']:
            frame = apply_broken_lines(frame, intensity=settings['broken_lines_intensity'])
        if settings['digital_grid']:
            frame = apply_digital_grid(frame, intensity=settings['digital_grid_intensity'])

        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        progress_bar.progress((i + 1) / n_frames)

# === GENERA VIDEO ===
def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

# === MAIN APP ===
def main():
    st.title(":camera: Glitch Video Studio")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)
    ratio_choice = st.sidebar.selectbox("Proporzione video", ["1:1", "9:16", "16:9"])
    ratio_options = {
        "1:1": 1.0,
        "9:16": 9/16,
        "16:9": 16/9,
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader(":game_die: Effetti")

    settings = {
        'pixel_shuffle': st.sidebar.checkbox("Pixel Shuffle", value=True),
        'pixel_shuffle_intensity': st.sidebar.slider("Intensità Pixel Shuffle", 1, 30, 10),

        'rgb_shift': st.sidebar.checkbox("RGB Shift", value=True),
        'rgb_shift_intensity': st.sidebar.slider("Intensità RGB Shift", 1, 20, 10),

        'invert': st.sidebar.checkbox("Color Inversion", value=False),
        'invert_intensity': st.sidebar.slider("Intensità Color Inversion", 0, 1, 0, step=0.1),

        'noise': st.sidebar.checkbox("Analog Noise + Grain", value=False),
        'noise_intensity': st.sidebar.slider("Intensità Analog Noise", 0.0, 1.0, 0.2),

        'scanlines': st.sidebar.checkbox("Scanlines CRT", value=False),
        'scanlines_intensity': st.sidebar.slider("Intensità Scanlines", 0.0, 1.0, 0.3),

        'posterize': st.sidebar.checkbox("Posterize + Contrast", value=False),
        'posterize_levels': st.sidebar.slider("Livelli Posterize", 2, 10, 4),

        'pixel_stretch': st.sidebar.checkbox("Pixel Stretch", value=False),
        'pixel_stretch_intensity': st.sidebar.slider("Intensità Pixel Stretch", 1, 20, 5),

        'row_column_shift': st.sidebar.checkbox("Row/Column Shift", value=False),
        'row_column_shift_intensity': st.sidebar.slider("Intensità Row/Column Shift", 1, 20, 5),

        'edge_overlay': st.sidebar.checkbox("Edge Detection Overlay", value=False),
        'edge_overlay_intensity': st.sidebar.slider("Intensità Edge Overlay", 0.0, 1.0, 0.2),

        'hue_shift': st.sidebar.checkbox("Colori Psichedelici", value=False),
        'hue_shift_intensity': st.sidebar.slider("Intensità Hue Shift", 1, 180, 30),

        'band_noise': st.sidebar.checkbox("Disturbo a Bande Colorate", value=False),
        'band_noise_intensity': st.sidebar.slider("Intensità Disturbo a Bande", 0.0, 1.0, 0.3),

        'broken_lines': st.sidebar.checkbox("Linee Spezzate B/N + Colorate", value=False),
        'broken_lines_intensity': st.sidebar.slider("Intensità Linee Spezzate", 0.0, 1.0, 0.3),

        'digital_grid': st.sidebar.checkbox("Griglia Digitale", value=False),
        'digital_grid_intensity': st.sidebar.slider("Intensità Griglia Digitale", 0.0, 1.0, 0.3),

        # Movimento organico 3D
        'motion_angle': st.sidebar.slider("Movimento 3D Angolo (gradi)", 0, 5, 2),
        'motion_zoom': st.sidebar.slider("Movimento 3D Zoom (%)", 0, 5, 2),
        'motion_tx': st.sidebar.slider("Movimento 3D Traslazione X (px)", 0, 10, 5),
        'motion_ty': st.sidebar.slider("Movimento 3D Traslazione Y (px)", 0, 10, 5),
    }

    generate_btn = st.sidebar.button(":clapper: Genera Video")

    if not uploaded_img:
        st.warning("Carica un'immagine per iniziare")
        return

    img = Image.open(uploaded_img).convert('RGB')
    img_np = np.array(img)

    # Ridimensiona e pad per proporzione scelta
    img_np = resize_and_pad(img_np, ratio_options[ratio_choice])

    st.image(img_np, caption="Immagine caricata ridimensionata", use_container_width=True)

    if generate_btn:
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

if __name__ == "__main__":
    main()
