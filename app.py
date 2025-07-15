import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import subprocess
import os
from pathlib import Path

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
        
        available_h = h - ty
        available_w = w - tx
        
        block_to_put = block[:available_h, :available_w]

        new_frame[ty:ty+block_to_put.shape[0], tx:tx+block_to_put.shape[1]] = block_to_put
    return new_frame

def apply_rgb_shift(frame, intensity=5):
    b, g, r = cv2.split(frame)
    rows, cols = frame.shape[:2]
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

def resize_and_pad(img, target_ratio):
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = w
        new_h = int(w / target_ratio)
    else:
        new_h = h
        new_w = int(h * target_ratio)
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
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
    filters = []
    if glitch_effect == "Color Inversion":
        filters.append("negate")
    elif glitch_effect == "Analog Noise":
        filters.append("noise=alls=20:allf=t")
    elif glitch_effect == "Scanlines VHS":
        filters.append("drawbox=y=0:color=black@0.3:width=iw:height=4:t=fill")
    vf_filter = ",".join(filters) if filters else "null"
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
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error("❌ Errore FFmpeg:")
        st.code(e.stderr)
        return False

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
    aspect_ratio = get_aspect_ratio(resolution)
    img_np_resized = resize_and_pad(img_np, aspect_ratio)

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

    audio_path = None
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix) as f:
            f.write(uploaded_audio.read())
            audio_path = f.name

    if st.button("Genera Video Glitch"):
        with st.spinner("Generazione video..."):
            temp_dir = tempfile.TemporaryDirectory()
            img_path = Path(temp_dir.name) / "input.jpg"
            output_path = Path(temp_dir.name) / "output.mp4"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np_resized, cv2.COLOR_RGB2BGR))
            progress_bar = st.progress(0)
            success = generate_video_with_ffmpeg(img_path, audio_path, output_path, duration, 30,
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
