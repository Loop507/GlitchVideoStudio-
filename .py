# streamlit_app.py
# Copyright (c) 2025 Loop507
# MIT License - https://opensource.org/licenses/MIT 

import streamlit as st
import numpy as np
import cv2
import random
import tempfile
import os

st.set_page_config(page_title="Glitch Video Studio", layout="centered")

st.title("🎞️ Glitch Video Studio")
st.markdown("by **Loop507** – Carica un'immagine e genera un video glitchato direttamente nel browser.")

# --- Effetti glitch ---
def apply_pixel_shuffle(frame, intensity=5):
    height, width = frame.shape[:2]
    block_size = intensity
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = frame[y:y+block_size, x:x+block_size]
            if block.shape[0] and block.shape[1]:
                blocks.append((x, y, block))
    random.shuffle(blocks)
    new_frame = np.zeros_like(frame)
    for i, (x, y, block) in enumerate(blocks):
        idx = random.randint(0, len(blocks)-1)
        tx, ty, _ = blocks[idx]
        new_frame[ty:ty+block_size, tx:tx+block_size] = block
    return new_frame

def apply_color_shift(frame, intensity=20):
    b, g, r = cv2.split(frame)
    b = np.roll(b, shift=random.randint(-intensity, intensity), axis=1)
    r = np.roll(r, shift=random.randint(-intensity, intensity), axis=1)
    return cv2.merge([b, g, r])

def apply_scanlines(frame, intensity=1):
    height, width = frame.shape[:2]
    for y in range(0, height, random.randint(2, 5)):
        frame[y:y+1, :] = np.clip(frame[y:y+1, :] - random.randint(20, 50), 0, 255)
    return frame

def apply_vhs_noise(frame, intensity=5):
    noise = np.random.randint(0, 255, (frame.shape[0], frame.shape[1]), dtype=np.uint8)
    _, mask = cv2.threshold(noise, 230, 255, cv2.THRESH_BINARY)
    white_noise = np.zeros_like(frame)
    white_noise[mask == 255] = [255, 255, 255]
    return cv2.addWeighted(frame, 0.8, white_noise, 0.2, 0)

# --- Generatore video glitch ---
def create_glitch_video(image, duration=5, fps=30):
    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, "output_video.mp4")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    num_frames = fps * duration

    for i in range(num_frames):
        frame = image.copy()
        if i % 2 == 0:
            frame = apply_pixel_shuffle(frame, intensity=5)
        if i % 5 == 0:
            frame = apply_color_shift(frame, intensity=20)
        if i % 7 == 0:
            frame = apply_scanlines(frame)
        if i % 10 == 0:
            frame = apply_vhs_noise(frame)
        out.write(frame)

    out.release()
    return video_path, temp_dir

# --- Interfaccia Streamlit ---
uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Immagine caricata", use_column_width=True)

    duration = st.slider("Durata del video (secondi)", 1, 30, 5)
    if st.button("Crea Video Glitch"):
        with st.spinner("Generando video..."):
            video_path, temp_dir = create_glitch_video(img, duration=duration)

            with open(video_path, "rb") as f:
                st.success("Fatto! Clicca per scaricare.")
                st.download_button("📥 Scarica il video glitchato", f, file_name="glitch_video.mp4", mime="video/mp4")

        temp_dir.cleanup()
