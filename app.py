import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import subprocess
import os
from pathlib import Path
from pydub import AudioSegment

# Funzioni glitch base (più avanti si possono ampliare)

def apply_pixel_shuffle(frame, intensity=5):
    h, w = frame.shape[:2]
    bs = max(1, intensity)
    blocks = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            block = frame[y:y+bs, x:x+bs]
            if block.shape[0] > 0 and block.shape[1] > 0:
                blocks.append((x, y, block))
    np.random.shuffle(blocks)
    new_frame = np.zeros_like(frame)
    for i, (ox, oy, block) in enumerate(blocks):
        idx = np.random.randint(len(blocks))
        tx, ty, _ = blocks[idx]
        bh, bw = block.shape[:2]
        new_frame[ty:ty+bh, tx:tx+bw] = block
    return new_frame

def apply_rgb_shift(frame, intensity=5):
    b, g, r = cv2.split(frame)
    rows, cols = frame.shape[:2]
    # sposta canali RGB di random offset max intensity
    def shift_channel(ch):
        dx = np.random.randint(-intensity, intensity)
        dy = np.random.randint(-intensity, intensity)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(ch, M, (cols, rows))
    r_s = shift_channel(r)
    g_s = shift_channel(g)
    b_s = shift_channel(b)
    return cv2.merge([b_s, g_s, r_s])

def apply_color_inversion(frame):
    return 255 - frame

def apply_analog_noise(frame, intensity=10):
    noise = np.random.normal(0, intensity, frame.shape).astype(np.uint8)
    noised = cv2.add(frame, noise)
    return noised

def apply_scanlines(frame, intensity=5):
    h, w = frame.shape[:2]
    for y in range(0, h, intensity*2):
        frame[y:y+intensity, :] = frame[y:y+intensity, :] * 0.5
    return frame

# Funzione ridimensiona + padding per risoluzioni
def resize_and_pad(img, target_ratio):
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)
    # ridimensiona immagine alla nuova dimensione con padding nero
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    # calcola offset per centrare
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    result[y_off:y_off+h, x_off:x_off+w] = img
    return result

def get_aspect_ratio(res):
    if res == "1:1":
        return 1.0
    elif res == "9:16":
        return 9/16
    elif res == "16:9":
        return 16/9

def generate_video_with_ffmpeg(img_path, audio_path, output_path, duration, fps, glitch_effect, intensity, progress_bar):
    # Costruiamo il filtro ffmpeg in base all'effetto scelto
    # (Es: pixel shuffle dinamico fatto in python, qui invece solo effetti analogici e glitch visivi via ffmpeg)
    filters = []
    # esempio effetti ffmpeg semplici da aggiungere:
    if glitch_effect == "Color Inversion":
        filters.append("negate")
    elif glitch_effect == "Analog Noise":
        filters.append("noise=alls=20:allf=t")
    elif glitch_effect == "Scanlines":
        filters.append("drawbox=y=0:color=black@0.3:width=iw:height=4:t=fill")
    # ...aggiungere altri filtri custom

    vf_filter = ",".join(filters) if filters else "null"

    # Costruzione comando ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(img_path),
    ]

    if audio_path:
        cmd += ["-i", str(audio_path), "-shortest"]

    cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-t", str(duration),
        "-r", str(fps),
        "-vf", vf_filter,
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    # esecuzione con aggiornamento progress bar (semplice, stima tempo)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        line = process.stderr.readline()
        if line == '' and process.poll() is not None:
            break
        if "frame=" in line:
            # estrai frame corrente e aggiorna barra
            parts = line.strip().split()
            for p in parts:
                if p.startswith("frame="):
                    frame_num = int(p.split("=")[1])
                    progress = min(frame_num / (duration*fps), 1.0)
                    progress_bar.progress(progress)
                    break
    process.wait()
    return process.returncode == 0

# Main App Streamlit

def main():
    st.title("🎥 Glitch Video Studio Minimal")
    st.sidebar.header("⚙️ Carica e Imposta")

    uploaded_image = st.sidebar.file_uploader("Carica immagine (obbligatorio)", type=["png", "jpg", "jpeg"])
    uploaded_audio = st.sidebar.file_uploader("Carica audio (opzionale)", type=["mp3", "wav"])

    resolution = st.sidebar.selectbox("Seleziona risoluzione", ["1:1", "9:16", "16:9"])
    glitch_effect = st.sidebar.selectbox("Seleziona effetto glitch",
                                         ["Pixel Shuffle", "RGB Shift", "Color Inversion", "Analog Noise", "Scanlines VHS"])

    duration = st.sidebar.slider("Durata video (secondi)", 1, 30, 5)
    intensity = st.sidebar.slider("Intensità glitch", 1, 10, 5)

    if not uploaded_image:
        st.warning("⚠️ Carica un'immagine per proseguire")
        return

    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)

    # Ridimensiona + padding immagine secondo risoluzione
    aspect_ratio = get_aspect_ratio(resolution)
    img_np_resized = resize_and_pad(img_np, aspect_ratio)

    # Preview glitch frame interattiva
    st.subheader("Anteprima glitch")
    preview = None
    if glitch_effect == "Pixel Shuffle":
        preview = apply_pixel_shuffle(img_np_resized.copy(), intensity=intensity)
    elif glitch_effect == "RGB Shift":
        preview = apply_rgb_shift(img_np_resized.copy(), intensity=intensity)
    elif glitch_effect == "Color Inversion":
        preview = apply_color_inversion(img_np_resized.copy())
    elif glitch_effect == "Analog Noise":
        preview = apply_analog_noise(img_np_resized.copy(), intensity=intensity*2)
    elif glitch_effect == "Scanlines VHS":
        preview = apply_scanlines(img_np_resized.copy(), intensity=intensity*2)
    if preview is not None:
        st.image(preview, use_column_width=True)

    # Determina durata video in base a audio o slider
    audio_path = None
    duration_final = duration
    if uploaded_audio:
        # salva audio temporaneo e calcola durata
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_bytes = uploaded_audio.read()
            f.write(audio_bytes)
            audio_path = f.name
        audio_seg = AudioSegment.from_file(audio_path)
        duration_final = audio_seg.duration_seconds
        # Stessa durata video audio

    if st.button("Genera Video Glitch"):
        with st.spinner("Generazione video..."):
            temp_dir = tempfile.TemporaryDirectory()
            img_path = Path(temp_dir.name) / "input.jpg"
            output_path = Path(temp_dir.name) / "output.mp4"

            # Salva immagine ridimensionata per ffmpeg
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2BGR))

            # Progress bar
            progress_bar = st.progress(0)

            success = generate_video_with_ffmpeg(img_path, audio_path, output_path, duration_final, 30,
                                                 glitch_effect, intensity, progress_bar)

            if success and output_path.exists():
                video_bytes = output_path.read_bytes()
                st.success("✅ Video generato!")
                st.video(video_bytes)
                st.download_button("⬇️ Scarica Video", data=video_bytes, file_name="glitch_video.mp4",
                                   mime="video/mp4")
            else:
                st.error("❌ Errore generazione video.")

            temp_dir.cleanup()
            if audio_path:
                os.unlink(audio_path)

if __name__ == "__main__":
    main()
