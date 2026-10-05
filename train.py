import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'plant_disease_model.h5'
CLASSES_SAVE_PATH = 'class_names.json'

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] The directory '{DATASET_DIR}' was not found!")
        return

    print("[INFO] Preparing high-accuracy dataset pipeline with data augmentation...")
    
    # Advanced data augmentation to simulate different rover camera angles, lighting, and distances
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    # Validation generator (only rescale, no random distortions)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save class names mapping
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"[INFO] Saved {len(class_names)} classes to {CLASSES_SAVE_PATH}")

    # Compute balanced class weights to give rarer diseases fair importance
    total_samples = len(train_generator.classes)
    num_classes = len(class_names)
    class_counts = np.bincount(train_generator.classes)
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = float(total_samples / (num_classes * max(class_counts[i], 1)))
    print(f"[INFO] Computed class weights: {class_weights}")

    print("[INFO] Building deep neural network architecture...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Phase 1: Freeze base model
    base_model.trainable = False

    # Enhanced classification head with BatchNormalization and Dropout to prevent overfitting
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\n[Phase 1/2] Training classification head for {INITIAL_EPOCHS} epochs...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=[reduce_lr]
    )

    print("\n[Phase 2/2] Fine-tuning convolutional layers for high precision...")
    # Unfreeze the top 40 layers of MobileNetV2 to fine-tune to eggplant leaf textures
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[reduce_lr]
    )

    print("\n[INFO] Saving high-accuracy trained model...")
    model.save(MODEL_SAVE_PATH)
    print(f"[SUCCESS] High-accuracy model saved as {MODEL_SAVE_PATH}!")

if __name__ == '__main__':
    train_model()
