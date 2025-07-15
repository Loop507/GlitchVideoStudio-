import streamlit as st
import numpy as np
import cv2
import subprocess
import tempfile
import random
from pathlib import Path
from PIL import Image, ImageFilter
import math
import json
import io

st.set_page_config(page_title="🎥 Glitch Video Studio Completo", page_icon="🎥", layout="wide")

# --- FUNZIONI DI MOVIMENTO AVANZATO ---

def smooth_random_wave(frame_idx, total_frames, freq=0.1, amplitude=5, seed=42):
    random.seed(seed)
    base = np.sin(2 * np.pi * freq * frame_idx / total_frames) * amplitude
    noise = (random.random() - 0.5) * amplitude * 0.3
    return base + noise

def apply_advanced_motion(frame, frame_idx, total_frames, layers=3, max_translation=10, max_rotation=5, max_zoom=0.05):
    h, w = frame.shape[:2]
    new_frame = np.zeros_like(frame)
    layer_height = h // layers
    for i in range(layers):
        y_start = i * layer_height
        y_end = (i + 1) * layer_height if i < layers - 1 else h
        layer = frame[y_start:y_end, :, :]

        tx = smooth_random_wave(frame_idx + i * 10, total_frames, freq=0.05 + i*0.02, amplitude=max_translation)
        ty = smooth_random_wave(frame_idx + i * 15, total_frames, freq=0.07 + i*0.03, amplitude=max_translation)
        angle = smooth_random_wave(frame_idx + i * 20, total_frames, freq=0.04 + i*0.01, amplitude=max_rotation)
        zoom = 1 + smooth_random_wave(frame_idx + i * 25, total_frames, freq=0.03 + i*0.01, amplitude=max_zoom)

        center = (layer.shape[1]//2, (y_end - y_start)//2)
        M = cv2.getRotationMatrix2D(center, angle, zoom)
        moved_layer = cv2.warpAffine(layer, M, (layer.shape[1], layer.shape[0]), borderMode=cv2.BORDER_REFLECT)

        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        moved_layer = cv2.warpAffine(moved_layer, M_trans, (layer.shape[1], layer.shape[0]), borderMode=cv2.BORDER_REFLECT)

        new_frame[y_start:y_end, :, :] = moved_layer

    return new_frame

def apply_wave_deformation(frame, frame_idx, total_frames, amplitude=5, frequency=20):
    h, w = frame.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            offset_x = amplitude * math.sin(2 * math.pi * (y / frequency + frame_idx / total_frames * 10))
            offset_y = amplitude * math.cos(2 * math.pi * (x / frequency + frame_idx / total_frames * 10))
            map_x[y, x] = np.clip(x + offset_x, 0, w - 1)
            map_y[y, x] = np.clip(y + offset_y, 0, h - 1)
    distorted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return distorted

# --- EFFETTI GLITCH COMPLETI ---

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

def apply_vhs_effect(frame):
    frame = apply_scanlines(frame)
    frame = apply_wave_distortion(frame)
    frame = apply_analog_noise(frame, 0.03)
    return frame

def apply_ascii_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    chars = np.asarray(list(' .:-=+*%@#'))
    scaled = cv2.resize(gray, (80, 45))
    indices = (scaled / 255 * (len(chars) - 1)).astype(np.uint8)
    ascii_img = np.ones_like(frame) * 255
    y0 = 20
    for i, row in enumerate(indices):
        line = "".join(chars[c] for c in row)
        y = y0 + i * 10
        cv2.putText(ascii_img, line, (5, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return ascii_img

# --- FORMATI VIDEO E GESTIONE IMMAGINE ---

def fit_image_to_aspect(img_pil, target_ratio, bg_blur=True):
    w, h = img_pil.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.01:
        # quasi uguale, nessuna modifica
        return img_pil
    if current_ratio > target_ratio:
        # immagine più larga, riduci larghezza e sfondo con blur
        new_w = int(h * target_ratio)
        crop_x = (w - new_w) // 2
        cropped = img_pil.crop((crop_x, 0, crop_x + new_w, h))
    else:
        # immagine più alta, riduci altezza
        new_h = int(w / target_ratio)
        crop_y = (h - new_h) // 2
        cropped = img_pil.crop((0, crop_y, w, crop_y + new_h))
    if bg_blur:
        blurred = img_pil.filter(ImageFilter.GaussianBlur(40)).resize(cropped.size)
        base = blurred.convert('RGB')
        base.paste(cropped, (0, 0))
        return base
    else:
        return cropped

# --- GENERAZIONE FRAMES ---

def generate_glitch_frames(img_np, n_frames, output_dir, settings):
    progress_bar = st.progress(0)
    total_frames = n_frames
    for i in range(n_frames):
        frame = img_np.copy()
        frame = apply_advanced_motion(frame, i, total_frames,
                                      layers=settings['motion_layers'],
                                      max_translation=settings['motion_translation'],
                                      max_rotation=settings['motion_rotation'],
                                      max_zoom=settings['motion_zoom'])
        frame = apply_wave_deformation(frame, i, total_frames,
                                      amplitude=settings['wave_amplitude'],
                                      frequency=settings['wave_frequency'])

        # Applicazione effetti glitch condizionali con intensità
        if settings['pixel_shuffle']:
            frame = apply_pixel_shuffle(frame, int(settings['pixel_shuffle_int']))
        if settings['rgb_shift']:
            frame = apply_rgb_shift(frame, int(settings['rgb_shift_int']))
        if settings['invert']:
            frame = apply_color_inversion(frame)
        if settings['noise']:
            frame = apply_analog_noise(frame, settings['noise_int'])
        if settings['scanlines']:
            frame = apply_scanlines(frame)
        if settings['posterize']:
            frame = apply_posterization(frame, max(2, int(settings['posterize_lvl'])))
        if settings['hue_shift']:
            frame = apply_hue_shift(frame, int(settings['hue_shift_val']))
        if settings['glitch_grid']:
            frame = apply_glitch_grid(frame)
        if settings.get('jpeg', False):
            frame = apply_jpeg_artifacts(frame)
        if settings.get('rowcol', False):
            frame = apply_row_column_shift(frame)
        if settings.get('wave', False):
            frame = apply_wave_distortion(frame)
        if settings.get('stretch', False):
            frame = apply_pixel_stretch(frame)
        if settings.get('edge', False):
            frame = apply_edge_overlay(frame)
        if settings.get('vhs', False):
            frame = apply_vhs_effect(frame)
        if settings.get('ascii', False):
            frame = apply_ascii_effect(frame)

        fname = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        progress_bar.progress((i + 1) / n_frames)

# --- GENERA VIDEO ---

def generate_video_from_frames(output_path, frame_rate, temp_dir, width, height):
    # Aggiungiamo filtro crop o padding per il formato scelto
    vf_filter = f"scale={width}:{height},setsar=1:1"
    cmd = [
        'ffmpeg', '-y', '-framerate', str(frame_rate),
        '-i', f'{temp_dir}/frame_%04d.jpg',
        '-vf', vf_filter,
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)

# --- SALVATAGGIO E CARICAMENTO PRESET ---

def save_preset(settings):
    preset_json = json.dumps(settings, indent=2)
    st.download_button("Salva preset", data=preset_json, file_name="preset_glitch.json", mime="application/json")

def load_preset():
    uploaded_preset = st.sidebar.file_uploader("Carica preset JSON", type=["json"])
    if uploaded_preset is not None:
        try:
            loaded = json.load(uploaded_preset)
            st.success("Preset caricato!")
            return loaded
        except Exception as e:
            st.error(f"Errore caricamento preset: {e}")
    return None

# --- MAIN APP ---

def main():
    st.title(":camera: Glitch Video Studio Completo")
    st.sidebar.header(":gear: Impostazioni")

    uploaded_img = st.sidebar.file_uploader("Carica immagine", type=["png", "jpg", "jpeg"])
    duration = st.sidebar.slider("Durata video (sec)", 1, 300, 10)
    fps = st.sidebar.slider("FPS", 10, 30, 15)

    # Formato video
    format_choice = st.sidebar.selectbox("Formato video", options=["1:1", "9:16", "16:9"])
    if format_choice == "1:1":
        target_ratio = 1.0
        out_width, out_height = 720, 720
    elif format_choice == "9:16":
        target_ratio = 9/16
        out_width, out_height = 540, 960
    else:
        target_ratio = 16/9
        out_width, out_height = 1280, 720

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Movimento immagine")
    motion_layers = st.sidebar.slider("Numero layer movimento", 1, 6, 3)
    motion_translation = st.sidebar.slider("Massima traslazione px", 0, 20, 10)
    motion_rotation = st.sidebar.slider("Massima rotazione (gradi)", 0.0, 10.0, 5.0)
    motion_zoom = st.sidebar.slider("Massimo zoom (%)", 0.0, 0.1, 0.05)
    wave_amplitude = st.sidebar.slider("Ampiezza deformazione onda", 0, 20, 5)
    wave_frequency = st.sidebar.slider("Frequenza deformazione onda", 1, 100, 20)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕹️ Effetti glitch")
    pixel_shuffle = st.sidebar.checkbox("Pixel Shuffle", value=True)
    pixel_shuffle_int = st.sidebar.slider("Intensità Pixel Shuffle", 1, 20, 10)

    rgb_shift = st.sidebar.checkbox("RGB Shift", value=True)
    rgb_shift_int = st.sidebar.slider("Intensità RGB Shift", 1, 20, 5)

    invert = st.sidebar.checkbox("Color Inversion", value=False)
    noise = st.sidebar.checkbox("Analog Noise + Grain", value=False)
    noise_int = st.sidebar.slider("Intensità Noise", 0.01, 1.0, 0.1)

    scanlines = st.sidebar.checkbox("Scanlines CRT", value=False)
    posterize = st.sidebar.checkbox("Posterize + Contrast", value=False)
    posterize_lvl = st.sidebar.slider("Livello Posterize", 2, 8, 4)

    hue_shift = st.sidebar.checkbox("Hue Shift Psichedelico", value=False)
    hue_shift_val = st.sidebar.slider("Valore Hue Shift", 0, 180, 30)

    glitch_grid = st.sidebar.checkbox("Glitch Grid Overlay", value=False)
    jpeg = st.sidebar.checkbox("JPEG Artifacts", value=False)
    rowcol = st.sidebar.checkbox("Row/Column Shift", value=False)
    wave = st.sidebar.checkbox("Wave Distortion", value=False)
    stretch = st.sidebar.checkbox("Pixel Stretch", value=False)
    edge = st.sidebar.checkbox("Edge Overlay", value=False)
    vhs = st.sidebar.checkbox("VHS Effect", value=False)
    ascii = st.sidebar.checkbox("ASCII Effect", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Controlli Globali")
    global_intensity = st.sidebar.slider("Intensità Globale", 0.1, 3.0, 1.0, 0.1)
    global_speed = st.sidebar.slider("Velocità Movimento", 0.1, 3.0, 1.0, 0.1)
    global_color = st.sidebar.slider("Saturazione Colore", 0.0, 2.0, 1.0, 0.1)

    # Salvataggio/Caricamento preset
    loaded_preset = load_preset()
    if loaded_preset:
        # Sovrascrivi settings da preset
        pixel_shuffle = loaded_preset.get('pixel_shuffle', pixel_shuffle)
        pixel_shuffle_int = loaded_preset.get('pixel_shuffle_int', pixel_shuffle_int)
        rgb_shift = loaded_preset.get('rgb_shift', rgb_shift)
        rgb_shift_int = loaded_preset.get('rgb_shift_int', rgb_shift_int)
        invert = loaded_preset.get('invert', invert)
        noise = loaded_preset.get('noise', noise)
        noise_int = loaded_preset.get('noise_int', noise_int)
        scanlines = loaded_preset.get('scanlines', scanlines)
        posterize = loaded_preset.get('posterize', posterize)
        posterize_lvl = loaded_preset.get('posterize_lvl', posterize_lvl)
        hue_shift = loaded_preset.get('hue_shift', hue_shift)
        hue_shift_val = loaded_preset.get('hue_shift_val', hue_shift_val)
        glitch_grid = loaded_preset.get('glitch_grid', glitch_grid)
        jpeg = loaded_preset.get('jpeg', jpeg)
        rowcol = loaded_preset.get('rowcol', rowcol)
        wave = loaded_preset.get('wave', wave)
        stretch = loaded_preset.get('stretch', stretch)
        edge = loaded_preset.get('edge', edge)
        vhs = loaded_preset.get('vhs', vhs)
        ascii = loaded_preset.get('ascii', ascii)

        motion_layers = loaded_preset.get('motion_layers', motion_layers)
        motion_translation = loaded_preset.get('motion_translation', motion_translation)
        motion_rotation = loaded_preset.get('motion_rotation', motion_rotation)
        motion_zoom = loaded_preset.get('motion_zoom', motion_zoom)
        wave_amplitude = loaded_preset.get('wave_amplitude', wave_amplitude)
        wave_frequency = loaded_preset.get('wave_frequency', wave_frequency)

        global_intensity = loaded_preset.get('global_intensity', global_intensity)
        global_speed = loaded_preset.get('global_speed', global_speed)
        global_color = loaded_preset.get('global_color', global_color)

    settings = {
        'pixel_shuffle': pixel_shuffle,
        'pixel_shuffle_int': max(1, int(pixel_shuffle_int * global_intensity)),
        'rgb_shift': rgb_shift,
        'rgb_shift_int': max(1, int(rgb_shift_int * global_intensity)),
        'invert': invert,
        'noise': noise,
        'noise_int': noise_int * global_intensity,
        'scanlines': scanlines,
        'posterize': posterize,
        'posterize_lvl': max(2, int(posterize_lvl / global_intensity)),
        'hue_shift': hue_shift,
        'hue_shift_val': int(hue_shift_val * global_intensity),
        'glitch_grid': glitch_grid,
        'jpeg': jpeg,
        'rowcol': rowcol,
        'wave': wave,
        'stretch': stretch,
        'edge': edge,
        'vhs': vhs,
        'ascii': ascii,
        'motion_layers': motion_layers,
        'motion_translation': motion_translation * global_speed,
        'motion_rotation': motion_rotation * global_speed,
        'motion_zoom': motion_zoom * global_speed,
        'wave_amplitude': wave_amplitude,
        'wave_frequency': wave_frequency,
        'global_intensity': global_intensity,
        'global_speed': global_speed,
        'global_color': global_color
    }

    if uploaded_img:
        img_pil = Image.open(uploaded_img).convert('RGB')
        img_pil = fit_image_to_aspect(img_pil, target_ratio)
        img_np = np.array(img_pil)

        st.image(img_np, caption="Immagine caricata adattata", use_container_width=True)

        if st.button(":clapper: Avvia generazione"):
            with st.spinner("Generazione video glitch in corso..."):
                temp_dir = tempfile.TemporaryDirectory()
                output_dir = Path(temp_dir.name)
                n_frames = int(duration * fps)

                generate_glitch_frames(img_np, n_frames, output_dir, settings)

                video_path = output_dir / "glitch_video.mp4"
                try:
                    generate_video_from_frames(video_path, fps, output_dir, out_width, out_height)
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

    save_preset(settings)

if __name__ == "__main__":
    main()
