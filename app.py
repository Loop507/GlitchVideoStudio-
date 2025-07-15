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

# --- Effetti glitch ---

def animate_image_random(img, max_shift=10, max_zoom=0.03, max_rotate=2):
    h, w = img.shape[:2]
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    scale = 1 + random.uniform(-max_zoom, max_zoom)
    angle = random.uniform(-max_rotate, max_rotate)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[:,2] += [tx, ty]
    moved = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return moved

def apply_pixel_shuffle(frame, intensity=10):
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

def apply_rgb_shift(frame, max_shift=8):
    h, w = frame.shape[:2]
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    b, g, r = cv2.split(frame)
    def shift(c, dx, dy):
        return cv2.warpAffine(c, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([shift(b, -shift_x, 0), shift(g, 0, shift_y), shift(r, shift_x, -shift_y)])

def apply_color_inversion(frame):
    return cv2.bitwise_not(frame)

def apply_analog_noise(frame, amount=0.05):
    noise = np.random.randn(*frame.shape) * 255 * amount
    noisy = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_scanlines(frame):
    for y in range(0, frame.shape[0], 3):
        frame[y:y+1, :] = frame[y:y+1, :] // 2
    return frame

def apply_posterization(frame, levels=6):
    div = 256 // levels
    return (frame // div * div).astype(np.uint8)

def apply_ascii_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    chars = np.asarray(list(' .:-=+*%@#'))
    scaled = cv2.resize(gray, (80, 45))
    indices = (scaled / 255 * (len(chars) - 1)).astype(np.uint8)
    ascii_img = np.ones_like(frame) * 255
    y0 = 20
    for i, line in enumerate("\n".join("".join(chars[c] for c in row) for row in indices).splitlines()):
        y = y0 + i * 10
        cv2.putText(ascii_img, line, (5, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
    return ascii_img

def apply_jpeg_artifacts(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    _, encimg = cv2.imencode('.jpg', frame, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_row_column_shift(frame):
    h, w = frame.shape[:2]
    new_frame = frame.copy()
    for _ in range(5):
        row = random.randint(0, h - 1)
        shift = random.randint(-20, 20)
        new_frame[row] = np.roll(new_frame[row], shift, axis=0)
    for _ in range(5):
        col = random.randint(0, w - 1)
        shift = random.randint(-20, 20)
        new_frame[:, col] = np.roll(new_frame[:, col], shift, axis=0)
    return new_frame

def apply_wave_distortion(frame):
    h, w = frame.shape[:2]
    new_frame = np.zeros_like(frame)
    for y in range(h):
        offset = int(10.0 * math.sin(2 * math.pi * y / 32))
        new_frame[y] = np.roll(frame[y], offset, axis=0)
    return new_frame

def apply_pixel_stretch(frame):
    h, w = frame.shape[:2]
    new_frame = frame.copy()
    for _ in range(5):
        y = random.randint(0, h - 1)
        start = random.randint(0, w - 10)
        length = random.randint(5, 15)
        if start + length < w:
            stretch_segment = np.repeat(new_frame[y, start:start+length], 2, axis=0)[:length]
            new_frame[y, start:start+length] = stretch_segment
    return new_frame

def apply_edge_overlay(frame):
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(frame, 0.7, edge_colored, 0.3, 0)

def apply_hue_shift(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + random.randint(20, 100)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_glitch_grid(frame):
    h, w = frame.shape[:2]
    grid = frame.copy()
    step = 15
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), (100, 255, 255), 1)
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), (100, 255, 255), 1)
    return cv2.addWeighted(frame, 0.9, grid, 0.1, 0)

def apply_color_bands(frame):
    h, w = frame.shape[:2]
    new_frame = frame.copy()
    band_height = 4
    for y in range(0, h, band_height):
        color_shift = np.random.randint(-50, 50, size=3)
        new_frame[y:y+band_height] = np.clip(new_frame[y:y+band_height] + color_shift, 0, 255)
    noise = np.random.randint(0, 50, (h, w, 3))
    new_frame = np.clip(new_frame + noise, 0, 255).astype(np.uint8)
    return new_frame

def apply_broken_lines(frame):
    h, w = frame.shape[:2]
    new_frame = frame.copy()
    for y in range(0, h, 8):
        if random.random() > 0.5:
            new_frame[y:y+2, random.randint(0, w-10):random.randint(10, w)] = 0
    for x in range(0, w, 12):
        if random.random() > 0.5:
            new_frame[random.randint(0, h-10):random.randint(10, h), x:x+2] = 255
    return new_frame

def apply_circuit_grid(frame):
    h, w = frame.shape[:2]
    new_frame = frame.copy()
    step = 20
    for y in range(0, h, step):
        cv2.line(new_frame, (0, y), (w, y), (180, 180, 180), 1)
    for x in range(0, w, step):
        cv2.line(new_frame, (x, 0), (x, h), (180, 180, 180), 1)
    for _ in range(5):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = x1 + random.randint(-20, 20)
        y2 = y1 + random.randint(-20, 20)
        cv2.line(new_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
    return new_frame

# --- Generazione frames ---

def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    frames = []
    for i in range(n_frames):
        frame = animate_image_random(img_np, max_shift=10, max_zoom=0.02, max_rotate=2)

        if settings.get('random_mode', False):
            all_effects = [
                apply_pixel_shuffle, apply_rgb_shift, apply_color_inversion, apply_analog_noise,
                apply_scanlines, apply_posterization, apply_ascii_effect, apply_jpeg_artifacts,
                apply_row_column_shift, apply_wave_distortion, apply_pixel_stretch,
                apply_edge_overlay, apply_hue_shift, apply_glitch_grid,
                apply_color_bands, apply_broken_lines, apply_circuit_grid
            ]
            random.shuffle(all_effects)
            effects_to_apply = all_effects[:random.randint(3, 7)]
            for ef in effects_to_apply:
                frame = ef(frame)
        else:
            if settings['pixel_shuffle']:
                frame = apply_pixel_shuffle(frame, intensity=random.randint(5, 25))
            if settings['rgb_shift']:
                frame = apply_rgb_shift(frame)
            if settings['invert']:
                frame = apply_color_inversion(frame)
            if settings['noise']:
                frame = apply_analog_noise(frame)
            if settings['scanlines']:
                frame = apply_scanlines(frame)
            if settings['posterize']:
                frame = apply_posterization(frame)
            if settings['ascii']:
                frame = apply_ascii_effect(frame)
            if settings['jpeg']:
                frame = apply_jpeg_artifacts(frame)
            if settings['row_col_shift']:
                frame = apply_row_column_shift(frame)
            if settings['wave_distortion']:
                frame = apply_wave_distortion(frame)
            if settings['pixel_stretch']:
                frame = apply_pixel_stretch(frame)
            if settings['edge_overlay']:
                frame = apply_edge_overlay(frame)
            if settings['hue_shift']:
                frame = apply_hue_shift(frame)
            if settings['glitch_grid']:
                frame = apply_glitch_grid(frame)
            if settings['color_bands']:
                frame = apply_color_bands(frame)
            if settings['broken_lines']:
                frame = apply_broken_lines(frame)
            if settings['circuit_grid']:
                frame = apply_circuit_grid(frame)

        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(frame)
        progress_bar.progress((i + 1) / n_frames)

    return frames

# --- Genera video da frames ---

def generate_video_from_frames(output_path, frame_rate, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

# --- Main App ---

def main():
    st.title(":camera: Glitch Video Studio")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)
    n_frames = duration * fps

    random_mode = st.sidebar.checkbox("Modalità glitch casuale", value=True)

    settings = {
        'random_mode': random_mode,
        'pixel_shuffle': st.sidebar.checkbox("Pixel Shuffle", value=True),
        'rgb_shift': st.sidebar.checkbox("RGB Shift", value=True),
        'invert': st.sidebar.checkbox("Color Inversion", value=False),
        'noise': st.sidebar.checkbox("Analog Noise", value=True),
        'scanlines': st.sidebar.checkbox("Scanlines", value=False),
        'posterize': st.sidebar.checkbox("Posterization", value=False),
        'ascii': st.sidebar.checkbox("ASCII Art", value=False),
        'jpeg': st.sidebar.checkbox("JPEG Artifacts", value=False),
        'row_col_shift': st.sidebar.checkbox("Row/Column Shift", value=False),
        'wave_distortion': st.sidebar.checkbox("Wave Distortion", value=False),
        'pixel_stretch': st.sidebar.checkbox("Pixel Stretch", value=False),
        'edge_overlay': st.sidebar.checkbox("Edge Overlay", value=False),
        'hue_shift': st.sidebar.checkbox("Hue Shift", value=False),
        'glitch_grid': st.sidebar.checkbox("Glitch Grid", value=False),
        'color_bands': st.sidebar.checkbox("Color Bands", value=False),
        'broken_lines': st.sidebar.checkbox("Broken Lines", value=False),
        'circuit_grid': st.sidebar.checkbox("Circuit Grid", value=False),
    }

    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert('RGB')
        img_np = np.array(img)

        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            st.write("Generazione video in corso...")
            frames = generate_glitch_frames(img_np, n_frames, temp_path, settings)

            video_path = temp_path / "output.mp4"
            generate_video_from_frames(video_path, fps, temp_path)

            video_file = open(video_path, 'rb').read()
            st.video(video_file)

if __name__ == "__main__":
    main()
