import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

def create_model():
    # Load the pre-trained MobileNetV2 model, excluding the top (classification) layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base_model so its weights are not updated during training
    base_model.trainable = False

    # Add custom classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(dataset_dir='dataset', epochs=5):
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found. Please create it and add 'cats' and 'dogs' subfolders.")
        return

    # Use MobileNetV2's specific preprocessing function
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    
    print("Starting Transfer Learning training with MobileNetV2...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    
    model.save('cat_dog_pretrained_model.h5')
    print("Model saved as 'cat_dog_pretrained_model.h5'")

if __name__ == '__main__':
    # Ensure the dataset path exists
    os.makedirs('dataset/cats', exist_ok=True)
    os.makedirs('dataset/dogs', exist_ok=True)
    print("Place your training images into 'dataset/cats' and 'dataset/dogs', then run this script.")
    train_model(epochs=5)
