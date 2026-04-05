"""
Cats vs Dogs - Train on YOUR uploaded photos + Live Camera Recognition
=====================================================================
This script:
  1. Trains a MobileNetV2 model on your datasets/Cat and datasets/Dog photos
  2. Immediately launches live webcam recognition

Run:  python train_and_predict.py
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
import sys

# ─── CONFIG ─────────────────────────────────────────────────────
DATASET_DIR  = 'datasets'      # Your uploaded photos: datasets/Cat  &  datasets/Dog
MODEL_PATH   = 'cat_dog_pretrained_model.h5'
EPOCHS       = 5
BATCH_SIZE   = 32
IMG_SIZE     = (224, 224)
# ────────────────────────────────────────────────────────────────


def create_model():
    """Build MobileNetV2 transfer-learning model for binary cat/dog classification."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pretrained layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    """Train on datasets/Cat and datasets/Dog."""
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory '{DATASET_DIR}' not found!")
        sys.exit(1)

    # Check subfolders
    subfolders = os.listdir(DATASET_DIR)
    print(f"\n Found subfolders in '{DATASET_DIR}': {subfolders}")
    for sf in subfolders:
        p = os.path.join(DATASET_DIR, sf)
        if os.path.isdir(p):
            count = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
            print(f"   -> {sf}: {count} images")

    # Data generators with MobileNetV2 preprocessing
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=0.2
    )

    print("\n Loading training data...")
    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    print(" Loading validation data...")
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Show class mapping
    print(f"\n  Class mapping: {train_gen.class_indices}")
    print(f"   Training samples  : {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")

    # Build and train
    model = create_model()
    print(f"\n Training MobileNetV2 for {EPOCHS} epochs...\n")

    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        )
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # Save
    model.save(MODEL_PATH)
    print(f"\n Model saved as '{MODEL_PATH}'")

    # Print final metrics
    val_acc = history.history.get('val_accuracy', [0])[-1]
    val_loss = history.history.get('val_loss', [0])[-1]
    print(f"   Final val_accuracy : {val_acc:.4f}")
    print(f"   Final val_loss     : {val_loss:.4f}")

    return model


def run_live_camera(model=None):
    """Live webcam cat/dog recognition."""
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model '{MODEL_PATH}' not found. Train first!")
            sys.exit(1)
        print(" Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    print("\n Opening webcam... Press 'Q' to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Colors
    CAT_COLOR  = (255, 165, 0)   # Orange (BGR)
    DOG_COLOR  = (0, 200, 100)   # Green  (BGR)
    BG_COLOR   = (20, 20, 20)

    frame_count = 0
    prediction_interval = 5  # Predict every N frames for performance
    label = "Analyzing..."
    color = (200, 200, 200)
    score = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # Run inference every N frames
        if frame_count % prediction_interval == 0:
            # FIX: Convert BGR (OpenCV) to RGB (Model expectation)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            prediction = model.predict(img_array, verbose=0)
            score = float(prediction[0][0])

            if score > 0.5:
                label = f"DOG  ({score:.0%})"
                color = DOG_COLOR
            else:
                label = f"CAT  ({(1-score):.0%})"
                color = CAT_COLOR

        # ─── Draw HUD ───
        h, w = frame.shape[:2]

        # Semi-transparent top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Confidence bar
        bar_w = int(w * score)
        cv2.rectangle(frame, (0, 62), (bar_w, 70), DOG_COLOR, -1)
        cv2.rectangle(frame, (bar_w, 62), (w, 70), CAT_COLOR, -1)

        # Prediction label
        cv2.putText(frame, label, (15, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)

        # Side labels on confidence bar
        cv2.putText(frame, "CAT", (5, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "DOG", (w - 30, 69),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom info bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 35), (w, h), BG_COLOR, -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Press Q to quit | Trained on YOUR photos",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # Colored border based on prediction
        border_color = color
        cv2.rectangle(frame, (0, 70), (w-1, h-35), border_color, 2)

        cv2.imshow("Cats vs Dogs - Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Camera closed.")


# ─── MAIN ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("   CATS vs DOGS - Train & Live Recognition")
    print("=" * 60)

    # Step 1: Train on your uploaded photos
    print("\n STEP 1: Training model on your uploaded photos...")
    model = train_model()

    # Step 2: Launch live camera
    print("\n STEP 2: Starting live camera recognition...")
    run_live_camera(model)
