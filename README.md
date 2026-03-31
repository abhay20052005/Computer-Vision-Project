# Cats vs Dogs — Pretrained Model Recognition Project

> Powered by MobileNetV2 Transfer Learning

---

## File Overview

| File | Purpose |
|---|---|
| `requirements.txt` | Python library dependencies |
| `download_dataset.py` | Auto-downloads a sample dataset (~68 MB) |
| `train.py` | Fine-tunes MobileNetV2 on your dataset |
| `predict.py` | Classify a single image from disk |
| `live_predict.py` | Real-time webcam classification |

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2: Get a Dataset

**Option A – Auto-download (easiest)**
```bash
python download_dataset.py
```
This downloads the standard Google "cats and dogs filtered" dataset
(~1000 cats + ~1000 dogs) and organizes it automatically.

**Option B – Manual (Kaggle)**
Download from: https://www.kaggle.com/c/dogs-vs-cats/data  
Place images here:
```
dataset/
    cats/   ← put cat images here
    dogs/   ← put dog images here
```

---

## Step 3: Train the Model

```bash
python train.py
```
- Uses **MobileNetV2** (pre-trained on ImageNet) via Transfer Learning
- Only 5 epochs needed — expects **>90% accuracy**
- Saves model to `cat_dog_pretrained_model.h5`

---

## Step 4: Predict an Image

```bash
python predict.py path/to/some/image.jpg
```

---

## Step 5: Live Webcam Recognition

```bash
python live_predict.py
```
- Opens your webcam and classifies what it sees in real time
- A confidence bar is shown at the top of the window
- Press **Q** to quit

---

## How Transfer Learning Works

Instead of training a whole new CNN, we borrow **MobileNetV2** — a compact
but powerful model already trained on 1.4 million images.

```
[MobileNetV2 base (frozen)] → [GlobalAveragePooling] → [Dropout] → [Dense sigmoid]
         ↑                                                                   ↑
   ImageNet weights                                             Our new Cat/Dog head
   (not updated)                                                  (trained by us)
```
