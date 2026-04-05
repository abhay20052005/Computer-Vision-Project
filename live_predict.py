"""
Live Cats vs Dogs recognition using your webcam.
Requires a trained model ('cat_dog_pretrained_model.h5').
Train the model first: python train.py
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import sys

MODEL_PATH = 'cat_dog_pretrained_model.h5'

def run_live_recognition():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found.")
        print("Please train the model first: python train.py")
        sys.exit(1)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded! Opening webcam... Press 'Q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    
    # Colors for display
    CAT_COLOR = (255, 165, 0)   # Orange
    DOG_COLOR = (0, 200, 100)   # Green
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # FIX: Convert BGR (From camera) to RGB (What model expects)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))

        # Run prediction
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])

        if score > 0.5:
            label = f"Dog  ({score:.0%})"
            color = DOG_COLOR
        else:
            label = f"Cat  ({(1-score):.0%})"
            color = CAT_COLOR

        # Draw a rounded rectangle banner at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw a confidence bar
        bar_width = int(frame.shape[1] * score)
        cv2.rectangle(frame, (0, 55), (bar_width, 60), DOG_COLOR, -1)
        cv2.rectangle(frame, (bar_width, 55), (frame.shape[1], 60), CAT_COLOR, -1)

        # Draw label text
        cv2.putText(frame, label, (15, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

        # Draw quit hint
        cv2.putText(frame, "Press Q to quit", (frame.shape[1] - 190, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Cats vs Dogs - Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_recognition()
