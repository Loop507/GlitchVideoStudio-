import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import random
import json
from pathlib import Path
from PIL import Image
import math
import io

st.set_page_config(page_title="🎥 Glitch Video Studio", page_icon="🎥", layout="wide")

# --- Effetti glitch esistenti + nuovi ---

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

def apply_ascii_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    chars = np.asarray(list(' .:-=+*%@#'))
    scaled = cv2.resize(gray, (80, 45))
    indices = (scaled / 255 * (len(chars) - 1)).astype(np.uint8)
    ascii_art = "\n".join("".join(chars[c] for c in row) for row in indices)
    ascii_img = np.ones_like(frame) * 255
    y0 = 20
    for i, line in enumerate(ascii_art.splitlines()):
        y = y0 + i * 10
        cv2.putText(ascii_img, line, (5, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
    return ascii_img

def apply_jpeg_artifacts(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
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
        offset = int(10.0 * math.sin(2 * math.pi * y / 64))
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

def apply_frame_skip_duplicate(frames, skip_prob=0.1, duplicate_prob=0.1):
    out_frames = []
    i = 0
    while i < len(frames):
        if random.random() < skip_prob:
            i += 1  # skip frame
            continue
        out_frames.append(frames[i])
        if random.random() < duplicate_prob:
            out_frames.append(frames[i])  # duplicate frame
        i += 1
    return out_frames

def apply_band_noise(frame, intensity=0.2):
    h, w = frame.shape[:2]
    noisy = frame.copy()
    for y in range(h):
        if random.random() < intensity:
            color = [random.randint(100,255) for _ in range(3)]
            noisy[y] = color
    return noisy

def apply_broken_lines(frame, intensity=0.1):
    h, w = frame.shape[:2]
    broken = frame.copy()
    for _ in range(int(h*intensity)):
        y = random.randint(0, h-1)
        x_start = random.randint(0, w//2)
        length = random.randint(5, w//3)
        broken[y, x_start:x_start+length] = 0
    return broken

def apply_circuit_grid(frame):
    h, w = frame.shape[:2]
    grid = frame.copy()
    step = 15
    for y in range(0, h, step):
        color = (100, 100, 100)
        cv2.line(grid, (0, y), (w, y), color, 1)
    for x in range(0, w, step):
        color = (100, 100, 100)
        cv2.line(grid, (x, 0), (x, h), color, 1)
    return cv2.addWeighted(frame, 0.8, grid, 0.2, 0)

def apply_base_motion(frame, frame_idx, total_frames):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    angle = np.sin(2 * np.pi * frame_idx / total_frames) * 2
    scale = 1 + 0.01 * np.sin(2 * np.pi * frame_idx / total_frames)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    moved = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return moved

# --- Funzione per ridimensionare + crop o sfondo sfocato
def resize_and_pad(img, target_ratio, method='crop'):
    h, w = img.shape[:2]
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.01:
        return img  # già proporzionato

    if method == 'crop':
        # Crop centrale
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            return img[:, start_x:start_x+new_w]
        else:
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            return img[start_y:start_y+new_h, :]
    else:
        # Sfondo sfocato (blurred)
        new_w, new_h = w, h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
        else:
            new_h = int(w / target_ratio)
        resized_img = cv2.resize(img, (new_w, new_h))
        # Sfondo sfocato
        bg = cv2.GaussianBlur(img, (51, 51), 30)
        pad_x1 = (w - new_w) // 2
        pad_y1 = (h - new_h) // 2
        bg[pad_y1:pad_y1+new_h, pad_x1:pad_x1+new_w] = resized_img
        return bg

# --- Genera frame con glitch e motion ---
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    frames = []
    for i in range(n_frames):
        base_frame = apply_base_motion(img_np.copy(), i, n_frames)
        frame = base_frame.copy()

        if settings['pixel_shuffle']: frame = apply_pixel_shuffle(frame, settings['pixel_shuffle_int'])
        if settings['rgb_shift']: frame = apply_rgb_shift(frame, settings['rgb_shift_int'])
        if settings['invert']: frame = apply_color_inversion(frame)
        if settings['noise']: frame = apply_analog_noise(frame, settings['noise_int'])
        if settings['scanlines']: frame = apply_scanlines(frame)
        if settings['posterize']: frame = apply_posterization(frame, settings['posterize_lvl'])
        if settings['hue_shift']: frame = apply_hue_shift(frame, settings['hue_shift_val'])
        if settings['glitch_grid']: frame = apply_glitch_grid(frame)
        if settings.get('jpeg'): frame = apply_jpeg_artifacts(frame)
        if settings.get('rowcol'): frame = apply_row_column_shift(frame)
        if settings.get('wave'): frame = apply_wave_distortion(frame)
        if settings.get('stretch'): frame = apply_pixel_stretch(frame)
        if settings.get('edge'): frame = apply_edge_overlay(frame)
        if settings.get('vhs'): frame = apply_vhs_effect(frame)
        if settings.get('band'): frame = apply_band_noise(frame, 0.2)
        if settings.get('broken'): frame = apply_broken_lines(frame, 0.1)
        if settings.get('circuit'): frame = apply_circuit_grid(frame)

        frames.append(frame)
        progress_bar.progress((i + 1) / n_frames)

    # Frame skip/duplicate (effetti fisici glitch)
    frames = apply_frame_skip_duplicate(frames, settings.get('frame_skip_prob', 0.0), settings.get('frame_dup_prob', 0.0))

    # Salva tutti i frame
    for i, f in enumerate(frames):
        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    return len(frames)

# --- Genera video da frames ---
def generate_video_from_frames(output_path, fps, temp_dir):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)

# --- Main App ---
def main():
    st.title(":camera: Glitch Video Studio")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)

    # Formati video
    format_options = {
        "1:1": 1.0,
        "9:16": 9/16,
        "16:9": 16/9
    }
    format_selected = st.sidebar.selectbox("Formato video", list(format_options.keys()))
    resize_method = st.sidebar.radio("Metodo adattamento formato", ["Crop centrale", "Sfondo sfocato"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader(":control_knobs: Controlli Globali")
    global_intensity = st.sidebar.slider("Intensità Globale", 0.1, 2.0, 1.0, 0.1)
    global_speed = st.sidebar.slider("Velocità Glitch", 0.1, 2.0, 1.0, 0.1)
    global_color = st.sidebar.slider("Saturazione Colore", 0.0, 2.0, 1.0, 0.1)

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
        'band': st.sidebar.checkbox("Disturbo a Bande Colorate", value=False),
        'broken': st.sidebar.checkbox("Linee Spezzate", value=False),
        'circuit': st.sidebar.checkbox("Griglia Circuiti", value=False),
        'frame_skip_prob': st.sidebar.slider("Probabilità Frame Skip", 0.0, 0.5, 0.0, 0.05),
        'frame_dup_prob': st.sidebar.slider("Probabilità Frame Duplicate", 0.0, 0.5, 0.0, 0.05)
    }

    # Pulsanti preset
    st.sidebar.markdown("---")
    st.sidebar.subheader("Preset")
    if st.sidebar.button("Salva preset"):
        preset_data = json.dumps(settings)
        st.sidebar.download_button("Download preset JSON", data=preset_data, file_name="preset_glitch.json")
    uploaded_preset = st.sidebar.file_uploader("Carica preset JSON", type=["json"])
    if uploaded_preset:
        preset_loaded = json.load(uploaded_preset)
        for k in settings.keys():
            if k in preset_loaded:
                settings[k] = preset_loaded[k]
        st.experimental_rerun()

    if uploaded_img:
        img = Image.open(uploaded_img).convert('RGB')
        img_np = np.array(img)

        # Adatta immagine al formato scelto
        img_np = resize_and_pad(img_np, format_options[format_selected], resize_method)

        # Applica saturazione colore globale
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= global_color
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        st.image(img_np, caption="Immagine adattata al formato video", use_container_width=True)

        if st.button(":clapper: Avvia generazione"):
            with st.spinner("Generazione glitch video..."):
                temp_dir = tempfile.TemporaryDirectory()
                output_dir = Path(temp_dir.name)
                n_frames = int(duration * fps * global_speed)

                n_actual_frames = generate_glitch_frames(img_np, n_frames, output_dir, settings)

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
