import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda, Softmax
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# ======================
# Data Preparation
# ======================

trial_num = 1

train_dir = './blinknet_data_set/dataset/'  # path where blink and non_blink folders are

train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    './blinknet_train_data/train/',
    target_size=(30, 30),
    batch_size=32,
    subset='training',
    shuffle=True,
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    './blinknet_train_data/val/',
    target_size=(30, 30),
    batch_size=32,
    subset='validation',
    shuffle=True,
    class_mode = 'categorical'
)

# ======================
# Compile & Train
# ======================


model = load_model("./blink_model_trained.h5")  # or "saved_model/"
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100
)

# ======================
# Save Model
# ======================

model.save(f"./blinknet_models/blink_model_trained_{trial_num}.h5")

