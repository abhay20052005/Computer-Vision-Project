"""
Generates a synthetic test dataset of cat-like and dog-like images
so you can immediately test the full training pipeline without downloading data.

Each class gets a distinct visual pattern:
  - Cats: orange/warm tones with circular shapes
  - Dogs: blue/cool tones with rectangular shapes

Run: python generate_test_data.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import random

NUM_IMAGES = 300   # per class (300 cats + 300 dogs = 600 total)
IMG_SIZE   = 224

def make_cat_image():
    """Warm tones + circular/oval shapes (simulates cat face structure)."""
    r = random.randint(180, 240)
    g = random.randint(100, 160)
    b = random.randint(20,  80)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Face circle
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    draw.ellipse([cx-70, cy-60, cx+70, cy+60], fill=(r-40, g-30, b+10))

    # Eyes
    draw.ellipse([cx-35, cy-20, cx-15, cy+5],  fill=(20, 20, 20))
    draw.ellipse([cx+15, cy-20, cx+35, cy+5],  fill=(20, 20, 20))

    # Nose
    draw.polygon([(cx, cy+10), (cx-8, cy+25), (cx+8, cy+25)], fill=(180, 60, 60))

    # Ears (triangles)
    draw.polygon([(cx-70, cy-60), (cx-45, cy-100), (cx-20, cy-60)], fill=(r-20, g-20, b))
    draw.polygon([(cx+20, cy-60), (cx+45, cy-100), (cx+70, cy-60)], fill=(r-20, g-20, b))

    # Add noise
    px = np.array(img)
    noise = np.random.randint(-15, 15, px.shape, dtype=np.int16)
    px = np.clip(px.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(px)


def make_dog_image():
    """Cool tones + wider/rounder face with floppy ear shapes."""
    r = random.randint(60,  120)
    g = random.randint(100, 160)
    b = random.randint(180, 240)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    # Wider face (dog faces are rounder/wider)
    draw.ellipse([cx-80, cy-55, cx+80, cy+70], fill=(r+30, g+20, b-40))

    # Eyes (larger, rounder)
    draw.ellipse([cx-40, cy-15, cx-10, cy+15], fill=(20, 20, 20))
    draw.ellipse([cx+10, cy-15, cx+40, cy+15], fill=(20, 20, 20))

    # Nose (wider)
    draw.ellipse([cx-20, cy+20, cx+20, cy+45], fill=(30, 30, 30))

    # Floppy ears (rectangles drooping down the sides)
    draw.rectangle([cx-100, cy-40, cx-75, cy+60], fill=(r+20, g+10, b-50))
    draw.rectangle([cx+75,  cy-40, cx+100, cy+60], fill=(r+20, g+10, b-50))

    # Add noise
    px = np.array(img)
    noise = np.random.randint(-15, 15, px.shape, dtype=np.int16)
    px = np.clip(px.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(px)


def generate(num_images=NUM_IMAGES):
    os.makedirs("dataset/cats", exist_ok=True)
    os.makedirs("dataset/dogs", exist_ok=True)

    print(f"Generating {num_images} cat images...")
    for i in range(num_images):
        img = make_cat_image()
        img.save(f"dataset/cats/cat_{i:04d}.jpg")
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{num_images} cats done")

    print(f"\nGenerating {num_images} dog images...")
    for i in range(num_images):
        img = make_dog_image()
        img.save(f"dataset/dogs/dog_{i:04d}.jpg")
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{num_images} dogs done")

    print(f"\nSynthetic dataset ready!")
    print(f"  Cats : {num_images} images  ->  dataset/cats/")
    print(f"  Dogs : {num_images} images  ->  dataset/dogs/")
    print("\nNow run:  python train.py")

if __name__ == '__main__':
    generate()
