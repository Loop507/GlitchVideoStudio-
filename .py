# GlitchVideoStudio.py
# Copyright (c) 2025 Loop507
# MIT License - https://opensource.org/licenses/MIT 

import cv2
import numpy as np
import random
import os

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

def create_glitch_video_from_image(image_path, output_path="output_video.mp4", duration=5, fps=30):
    base_frame = cv2.imread(image_path)
    if base_frame is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    height, width = base_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = fps * duration

    for i in range(num_frames):
        frame = base_frame.copy()
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
    print(f"Video glitch creato: {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Glitch Video Studio by Loop507 ===")
    image_path = input("Inserisci il percorso dell'immagine: ").strip()
    if not os.path.exists(image_path):
        print("[ERRORE] Immagine non trovata.")
        exit()

    try:
        duration = int(input("Durata del video (secondi): "))
    except ValueError:
        print("[ERRORE] Devi inserire un numero valido per la durata.")
        exit()

    output_name = input("Nome del video di output (es. mio_video.mp4): ").strip()
    if not output_name.endswith(".mp4"):
        output_name += ".mp4"

    print("\nGenerando video glitchato...\n")
    result = create_glitch_video_from_image(image_path=image_path, output_path=output_name, duration=duration)
    print(f"\n✨ Video completato: {result}")
