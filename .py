# GlitchVideoStudio.py
# Copyright (c) 2025 Loop507
# MIT License - https://opensource.org/licenses/MIT 
import cv2
import numpy as np
import random
import os
import sys

def apply_pixel_shuffle(frame, intensity=5):
    height, width = frame.shape[:2]
    block_size = max(1, intensity)  # Evita blocchi di dimensione 0
    blocks = []
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Calcola le dimensioni effettive del blocco
            end_y = min(y + block_size, height)
            end_x = min(x + block_size, width)
            block = frame[y:end_y, x:end_x]
            
            if block.shape[0] > 0 and block.shape[1] > 0:
                blocks.append((x, y, block))
    
    if not blocks:
        return frame
    
    random.shuffle(blocks)
    new_frame = np.zeros_like(frame)
    
    for i, (orig_x, orig_y, block) in enumerate(blocks):
        # Prendi un blocco casuale dalla lista
        idx = random.randint(0, len(blocks) - 1)
        target_x, target_y, _ = blocks[idx]
        
        # Calcola le dimensioni per evitare overflow
        block_h, block_w = block.shape[:2]
        end_y = min(target_y + block_h, height)
        end_x = min(target_x + block_w, width)
        
        # Adatta il blocco se necessario
        if end_y - target_y < block_h or end_x - target_x < block_w:
            block = block[:end_y - target_y, :end_x - target_x]
        
        new_frame[target_y:end_y, target_x:end_x] = block
    
    return new_frame

def apply_color_shift(frame, intensity=20):
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return frame
    
    b, g, r = cv2.split(frame)
    b = np.roll(b, shift=random.randint(-intensity, intensity), axis=1)
    r = np.roll(r, shift=random.randint(-intensity, intensity), axis=1)
    return cv2.merge([b, g, r])

def apply_scanlines(frame, intensity=1):
    height, width = frame.shape[:2]
    step = max(2, random.randint(2, 5))
    
    for y in range(0, height, step):
        if y < height:
            darkness = random.randint(20, 50)
            frame[y:min(y+1, height), :] = np.clip(
                frame[y:min(y+1, height), :].astype(np.int16) - darkness, 
                0, 255
            ).astype(np.uint8)
    
    return frame

def apply_vhs_noise(frame, intensity=5):
    height, width = frame.shape[:2]
    noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Correzione della sintassi per threshold
    _, mask = cv2.threshold(noise, 230, 255, cv2.THRESH_BINARY)
    
    white_noise = np.zeros_like(frame)
    if len(frame.shape) == 3:  # Immagine a colori
        white_noise[mask == 255] = [255, 255, 255]
    else:  # Immagine in scala di grigi
        white_noise[mask == 255] = 255
    
    return cv2.addWeighted(frame, 0.8, white_noise, 0.2, 0)

def create_glitch_video_from_image(image_path, output_path="output_video.mp4", duration=5, fps=30):
    # Verifica che il file esista
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    
    base_frame = cv2.imread(image_path)
    if base_frame is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    
    height, width = base_frame.shape[:2]
    
    # Usa un codec più compatibile
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Verifica che il writer sia stato inizializzato correttamente
    if not out.isOpened():
        raise RuntimeError(f"Impossibile creare il file video: {output_path}")
    
    num_frames = fps * duration
    print(f"Generando {num_frames} frame...")
    
    for i in range(num_frames):
        frame = base_frame.copy()
        
        # Applica gli effetti glitch in modo più controllato
        if i % 3 == 0:
            frame = apply_pixel_shuffle(frame, intensity=random.randint(3, 8))
        if i % 4 == 0:
            frame = apply_color_shift(frame, intensity=random.randint(10, 30))
        if i % 6 == 0:
            frame = apply_scanlines(frame)
        if i % 8 == 0:
            frame = apply_vhs_noise(frame)
        
        out.write(frame)
        
        # Mostra il progresso
        if i % 30 == 0:
            progress = (i / num_frames) * 100
            print(f"Progresso: {progress:.1f}%")
    
    out.release()
    print(f"Video glitch creato: {output_path}")
    return output_path

def main():
    print("=== Glitch Video Studio by Loop507 ===")
    
    # Input del percorso dell'immagine
    image_path = input("Inserisci il percorso dell'immagine: ").strip()
    
    # Rimuovi eventuali virgolette
    image_path = image_path.strip('"\'')
    
    if not os.path.exists(image_path):
        print("[ERRORE] Immagine non trovata.")
        sys.exit(1)
    
    # Input della durata
    try:
        duration = int(input("Durata del video (secondi): "))
        if duration <= 0:
            print("[ERRORE] La durata deve essere un numero positivo.")
            sys.exit(1)
    except ValueError:
        print("[ERRORE] Devi inserire un numero valido per la durata.")
        sys.exit(1)
    
    # Input del nome del file di output
    output_name = input("Nome del video di output (es. mio_video.mp4): ").strip()
    if not output_name.endswith(".mp4"):
        output_name += ".mp4"
    
    print("\nGenerando video glitchato...\n")
    
    try:
        result = create_glitch_video_from_image(
            image_path=image_path, 
            output_path=output_name, 
            duration=duration
        )
        print(f"\n✨ Video completato: {result}")
    except Exception as e:
        print(f"[ERRORE] Si è verificato un errore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
