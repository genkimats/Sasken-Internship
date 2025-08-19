import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda, Softmax
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt   # <-- added for plotting
from keras.models import load_model, Model
from keras.layers import Dropout



# ======================
# Data Preparation
# ======================

trial_num = 1

# train_dir = './blinknet_data_set/dataset/'  # path where blink and non_blink folders are

datagen = ImageDataGenerator(validation_split=0.2)

train_generator = datagen.flow_from_directory(
    './blinknet_data/dataset/',   # <-- put all images here
    target_size=(30, 30),
    batch_size=32,
    subset='training',
    shuffle=True,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    './blinknet_data/dataset/',   # <-- same directory
    target_size=(30, 30),
    batch_size=32,
    subset='validation',
    shuffle=True,
    class_mode='categorical'
)

# ======================
# Compile & Train
# ======================

model = load_model("./blinknet_models/blink_model_trained.h5")  # or "saved_model/"

x = model.input
y = x

for layer in model.layers:
    if layer.name == "pool1":
        y = layer(y)
        y = Dropout(0.3, name="dropout1")(y)  # after pool1
    elif layer.name == "pool2":
        y = layer(y)
        y = Dropout(0.3, name="dropout2")(y)  # after pool2
    elif layer.name == "pool3":
        y = layer(y)
        y = Dropout(0.3, name="dropout3")(y)  # after GAP before Dense
    else:
        y = layer(y)

model = Model(inputs=x, outputs=y)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',   # watch validation loss
    factor=0.5,           # reduce LR by half
    patience=5,           # wait 5 epochs without improvement
    min_lr=1e-6,          # don’t go below this LR
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=200,
    callbacks=[lr_scheduler]
)

model.summary()

# Plot validation accuracy in a separate subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Loss subplot
axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].set_title('Model Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

# Accuracy subplot
axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].set_title('Model Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"./blinknet_models/loss_acc_subplot_{trial_num}.png")
plt.show()

# ======================
# Save Model
# ======================

model.save(f"./blinknet_models/blink_model_trained_{trial_num}.h5")

