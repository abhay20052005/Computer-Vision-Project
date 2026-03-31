"""
Download Cats vs Dogs dataset using tensorflow_datasets (tfds).
This is the most reliable method — no direct URL needed.
Run: pip install tensorflow-datasets
"""
import os
import shutil
from PIL import Image

def download_with_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("tensorflow-datasets not installed. Run: pip install tensorflow-datasets")
        return

    print("Downloading Cats vs Dogs dataset via tensorflow-datasets...")
    print("(This may take a few minutes on first run)\n")

    ds, info = tfds.load(
        'cats_vs_dogs',
        split='train[:80%]',
        with_info=True,
        as_supervised=True,
        download=True
    )

    os.makedirs("dataset/cats", exist_ok=True)
    os.makedirs("dataset/dogs", exist_ok=True)

    cat_count = 0
    dog_count = 0

    print("Saving images to dataset folder...")
    for i, (img_tensor, label) in enumerate(ds):
        img = Image.fromarray(img_tensor.numpy())
        if label.numpy() == 0:  # Cat
            img.save(f"dataset/cats/cat_{i}.jpg")
            cat_count += 1
        else:  # Dog
            img.save(f"dataset/dogs/dog_{i}.jpg")
            dog_count += 1

        if i % 500 == 0:
            print(f"  Saved {i} images so far... (cats: {cat_count}, dogs: {dog_count})")

        # Stop after collecting 1000 of each for faster training
        if cat_count >= 1000 and dog_count >= 1000:
            break

    print(f"\nDataset ready!")
    print(f"  Cats : {cat_count} images  →  dataset/cats/")
    print(f"  Dogs : {dog_count} images  →  dataset/dogs/")
    print("\nYou can now run:  python train.py")

if __name__ == '__main__':
    download_with_tfds()
