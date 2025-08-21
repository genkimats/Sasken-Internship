import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda, Softmax
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt   # <-- added for plotting
from keras.models import load_model, Model
from keras.layers import Dropout

class AdaptiveLR(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor_up=1.1, factor_down=0.5, patience=3, min_lr=1e-6, max_lr=1e-3, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.factor_up = factor_up      # multiply LR by this if metric improves
        self.factor_down = factor_down  # multiply LR by this if metric does not improve
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.best = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        
        if current is None:
            return
        
        if current < self.best:  # metric improved
            self.best = current
            self.wait = 0
            new_lr = min(lr * self.factor_up, self.max_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved. Increasing LR from {lr:.6f} to {new_lr:.6f}")
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        else:  # metric did not improve
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(lr * self.factor_down, self.min_lr)
                if self.verbose > 0:
                    print(f"\nEpoch {epoch+1}: {self.monitor} did not improve for {self.patience} epochs. Reducing LR from {lr:.6f} to {new_lr:.6f}")
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0


class StopOnValAcc(tf.keras.callbacks.Callback):
    def __init__(self, target_acc=1.0):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc >= self.target_acc:
            print(f"\nReached {self.target_acc*100:.1f}% validation accuracy. Stopping training!")
            self.model.stop_training = True


# ======================
# Data Preparation
# ======================


# train_dir = './blinknet_data_set/dataset/'  # path where blink and non_blink folders are

datagen = ImageDataGenerator(validation_split=0.2)

train_generator = datagen.flow_from_directory(
    './blinknet_data/train/',   # <-- put all images here
    target_size=(30, 30),
    batch_size=32,
    subset='training',
    shuffle=True,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    './blinknet_data/train/',   # <-- same directory
    target_size=(30, 30),
    batch_size=32,
    subset='validation',
    shuffle=True,
    class_mode='categorical'
)

# ======================
# Compile & Train
# ======================

# Load pretrained model
model = load_model("./blinknet_models/blink_model_trained.h5")
trial_num = 5  # Increment trial number

x = model.input
y = x

# ======================
# Freeze layers for fine-tuning
# ======================


# for layer in model.layers:
#     # Freeze all conv and batchnorm layers except the last conv block
#     if ("conv3" not in layer.name and "bn3" not in layer.name) and ("dense" not in layer.name):
#         layer.trainable = False
#     else:
#         layer.trainable = True


# Rebuild model with additional dropout layers
# for layer in model.layers:
#     if layer.name == "pool1":
#         y = layer(y)
#         y = Dropout(0.2, name="dropout1")(y)
#     elif layer.name == "pool2":
#         y = layer(y)
#         y = Dropout(0.3, name="dropout2")(y)
#     elif layer.name == "pool3":
#         y = layer(y)
#         y = Dropout(0.4, name="dropout3")(y)
#     else:
#         y = layer(y)

# model = Model(inputs=x, outputs=y)

# Recompile after freezing layers
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler
adaptive_lr = AdaptiveLR(
    monitor='val_loss',
    factor_up=1.05,     # slowly increase LR if improving
    factor_down=0.5,    # decrease LR if stuck
    patience=5,
    min_lr=1e-20,
    max_lr=1e-14,
    verbose=1
)

stop_on_val_acc = StopOnValAcc(target_acc=1.0)

# Train only unfrozen layers
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    # callbacks=[adaptive_lr, stop_on_val_acc]
)

model.summary()
model.save(f"./blinknet_models/blink_model_finetuned_{trial_num}.h5")


import matplotlib.pyplot as plt

# Create subplots for loss and accuracy
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot Training & Validation Loss ---
axs[0].plot(history.history['loss'], label='Training Loss')
axs[0].plot(history.history['val_loss'], label='Validation Loss')
axs[0].set_title('Model Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

# --- Plot Training & Validation Accuracy ---
axs[1].plot(history.history['accuracy'], label='Training Accuracy')
axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[1].set_title('Model Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

# Improve layout & save figure
plt.tight_layout()
plt.savefig(f"./blinknet_models/loss_acc_subplot_{trial_num}.png")
plt.show()
