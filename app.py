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
    # Shift random rows horizontally
    for _ in range(5):
        row = random.randint(0, h - 2)
        shift = random.randint(-10, 10)
        frame[row] = np.roll(frame[row], shift, axis=0)
    # Shift random cols vertically
    for _ in range(5):
        col = random.randint(0, w - 2)
        shift = random.randint(-10, 10)
        frame[:, col] = np.roll(frame[:, col], shift, axis=0)
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
        x = random.randint(0, w - 1)
        stretched[y] = stretched[y, x]
    return stretched

def apply_edge_overlay(frame):
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(frame, 0.8, edge_colored, 0.2, 0)

def apply_hue_shift(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + random.randint(10, 100)) % 180
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

def apply_band_noise(frame):
    h, w = frame.shape[:2]
    frame = frame.copy()
    for _ in range(15):
        y = random.randint(0, h - 1)
        thickness = random.randint(1, 3)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.line(frame, (0, y), (w, y), color, thickness)
    return frame

def apply_broken_lines(frame):
    h, w = frame.shape[:2]
    frame = frame.copy()
    for _ in range(20):
        x = random.randint(0, w - 10)
        y = random.randint(0, h - 10)
        length = random.randint(5, 30)
        thickness = random.randint(1, 3)
        color = (255, 255, 255) if random.random() > 0.8 else (random.randint(0,255), 0, 0)
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
    return frame

def apply_circuit_grid(frame):
    h, w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grid = np.zeros_like(frame)
    step = 15
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), 150, 1)
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), 150, 1)
    grid_colored = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
    frame_color = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), 0.7, grid_colored, 0.3, 0)
    return frame_color

def apply_organic_movement(frame, frame_idx, max_shift=5, max_rot=2, max_zoom=0.05):
    h, w = frame.shape[:2]
    # Traslazioni leggere con sinusoidi per movimento organico
    dx = int(max_shift * math.sin(2 * math.pi * frame_idx / 30))
    dy = int(max_shift * math.cos(2 * math.pi * frame_idx / 25))
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(frame, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Rotazione leggera
    angle = max_rot * math.sin(2 * math.pi * frame_idx / 40)
    M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

    moved = cv2.warpAffine(moved, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Zoom leggero (scaling)
    zoom_factor = 1 + max_zoom * math.sin(2 * math.pi * frame_idx / 50)
    center = (w//2, h//2)
    M_zoom = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    moved = cv2.warpAffine(moved, M_zoom, (w, h), borderMode=cv2.BORDER_REFLECT)

    return moved

def blend_frames(base, overlay, alpha=0.5):
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

# --- GENERA FRAME ---
def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    for i in range(n_frames):
        # Movimento organico base
        base_frame = apply_organic_movement(img_np, i)

        glitch_frame = base_frame.copy()

        # Applicazione effetti glitch come layer sovrapposti
        overlays = []

        if settings.get('pixel_shuffle', False):
            overlays.append(apply_pixel_shuffle(base_frame, intensity=random.randint(5, 30)))
        if settings.get('rgb_shift', False):
            overlays.append(apply_rgb_shift(base_frame))
        if settings.get('invert', False):
            overlays.append(apply_color_inversion(base_frame))
        if settings.get('noise', False):
            overlays.append(apply_analog_noise(base_frame))
        if settings.get('scanlines', False):
            overlays.append(apply_scanlines(base_frame))
        if settings.get('posterize', False):
            overlays.append(apply_posterization(base_frame))
        if settings.get('ascii', False):
            overlays.append(apply_ascii_effect(base_frame))
        if settings.get('jpeg', False):
            overlays.append(apply_jpeg_artifacts(base_frame))
        if settings.get('row_col_shift', False):
            overlays.append(apply_row_column_shift(base_frame))
        if settings.get('wave_distortion', False):
            overlays.append(apply_wave_distortion(base_frame))
        if settings.get('pixel_stretch', False):
            overlays.append(apply_pixel_stretch(base_frame))
        if settings.get('edge_overlay', False):
            overlays.append(apply_edge_overlay(base_frame))
        if settings.get('hue_shift', False):
            overlays.append(apply_hue_shift(base_frame))
        if settings.get('glitch_grid', False):
            overlays.append(apply_glitch_grid(base_frame))
        if settings.get('band_noise', False):
            overlays.append(apply_band_noise(base_frame))
        if settings.get('broken_lines', False):
            overlays.append(apply_broken_lines(base_frame))
        if settings.get('circuit_grid', False):
            overlays.append(apply_circuit_grid(base_frame))

        # Mischia overlay e fonde
        random.shuffle(overlays)
        for ov in overlays:
            glitch_frame = blend_frames(glitch_frame, ov, alpha=0.4)

        # Salva frame
        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(glitch_frame, cv2.COLOR_RGB2BGR))
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

# --- MAIN APP ---
def main():
    st.title(":camera: Glitch Video Studio")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 20, 5)
    fps = st.sidebar.slider("FPS", 10, 30, 15)

    st.sidebar.markdown("---")
    st.sidebar.subheader(":game_die: Effetti")

    settings = {
        'pixel_shuffle': st.sidebar.checkbox("Pixel Shuffle", value=True),
        'rgb_shift': st.sidebar.checkbox("RGB Shift", value=True),
        'invert': st.sidebar.checkbox("Color Inversion", value=False),
        'noise': st.sidebar.checkbox("Analog Noise + Grain", value=False),
        'scanlines': st.sidebar.checkbox("Scanlines CRT", value=False),
        'posterize': st.sidebar.checkbox("Posterize + Contrast", value=False),
        'ascii': st.sidebar.checkbox("ASCII Art", value=False),
        'jpeg': st.sidebar.checkbox("JPEG Compression Artifacts", value=False),
        'row_col_shift': st.sidebar.checkbox("Row/Column Shift", value=False),
        'wave_distortion': st.sidebar.checkbox("Wave Distortion", value=False),
        'pixel_stretch': st.sidebar.checkbox("Pixel Stretch", value=False),
        'edge_overlay': st.sidebar.checkbox("Edge Detection Overlay", value=False),
        'hue_shift': st.sidebar.checkbox("Hue Shift / Colori Psichedelici", value=False),
        'glitch_grid': st.sidebar.checkbox("Glitch Grid Overlay", value=False),
        'band_noise': st.sidebar.checkbox("Disturbo a Bande Colorate", value=False),
        'broken_lines': st.sidebar.checkbox("Linee Spezzate Interferenza", value=False),
        'circuit_grid': st.sidebar.checkbox("Circuit Grid Digitale", value=False)
    }

    level = st.sidebar.radio("Difficoltà Glitch", ['Soft', 'Medium', 'Hard'])

    # Modifica parametri in base a livello
    if level == 'Soft':
        settings = {k: (v and random.random() < 0.3) for k, v in settings.items()}
    elif level == 'Medium':
        settings = {k: (v and random.random() < 0.6) for k, v in settings.items()}
    else:
        # Hard abilita quasi tutto
        settings = {k: v for k, v in settings.items()}

    generate_btn = st.sidebar.button(":clapper: Genera Video")

    if not uploaded_img:
        st.warning("Carica un'immagine per iniziare")
        return

    img = Image.open(uploaded_img).convert('RGB')
    img_np = np.array(img)
    st.image(img_np, caption="Immagine caricata", use_container_width=True)

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
