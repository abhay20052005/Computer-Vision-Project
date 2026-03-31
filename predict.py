import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import sys
import os

def predict_image(image_path, model_path='cat_dog_pretrained_model.h5'):
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first by running 'python train.py'.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image. MobileNetV2 uses 224x224 input size.
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Crucial: Must use MobileNetV2's specific preprocess_input
    x = preprocess_input(x)

    # Predict
    classes = model.predict(x, batch_size=10)
    
    # The image data generator alphabetical folder order usually assigns 0 to 'cats' and 1 to 'dogs'
    if classes[0][0] > 0.5:
        print(f"Result: is a dog (Confidence: {float(classes[0][0]):.2f})")
    else:
        print(f"Result: is a cat (Confidence: {1.0 - float(classes[0][0]):.2f})")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image.jpg>")
    else:
        predict_image(sys.argv[1])
