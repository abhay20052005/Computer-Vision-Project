"""
Organizes the Kaggle 'Dogs vs Cats' dataset into dataset/cats and dataset/dogs.
1. Download train.zip from https://www.kaggle.com/c/dogs-vs-cats/data
2. Extract it so you have a folder full of cat.X.jpg and dog.X.jpg files
3. Set KAGGLE_TRAIN_DIR below to point to that folder
4. Run: python organize_kaggle.py
"""
import os
import shutil

# ← CHANGE THIS to where you extracted the Kaggle train folder
KAGGLE_TRAIN_DIR = r"C:\Users\abhay\Downloads\train"

def organize():
    if not os.path.exists(KAGGLE_TRAIN_DIR):
        print(f"Folder not found: {KAGGLE_TRAIN_DIR}")
        print("Please update KAGGLE_TRAIN_DIR in this script.")
        return

    os.makedirs("dataset/cats", exist_ok=True)
    os.makedirs("dataset/dogs", exist_ok=True)

    files = os.listdir(KAGGLE_TRAIN_DIR)
    cat_count = dog_count = 0

    for fname in files:
        src = os.path.join(KAGGLE_TRAIN_DIR, fname)
        if fname.startswith("cat."):
            shutil.copy2(src, os.path.join("dataset", "cats", fname))
            cat_count += 1
        elif fname.startswith("dog."):
            shutil.copy2(src, os.path.join("dataset", "dogs", fname))
            dog_count += 1

    print(f"Done! Cats: {cat_count}, Dogs: {dog_count}")
    print("Now run: python train.py")

if __name__ == '__main__':
    organize()
