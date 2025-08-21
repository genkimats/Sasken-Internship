import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras
from keras.layers import Dropout


# ======================
# Load existing model
# ======================
base_model = keras.models.load_model("./blinknet_models/blink_model_trained.h5")


# ======================
# Rebuild model with tunable dropout
# ======================
def build_model(hp):
    base_model = keras.models.load_model("./blinknet_models/blink_model_trained.h5")

    x = base_model.input
    y = base_model.layers[1](x)   # bias_subtract
    y = base_model.layers[2](y)   # conv1
    y = base_model.layers[3](y)   # conv1_bn
    y = base_model.layers[4](y)   # relu1
    y = base_model.layers[5](y)   # pool1
    y = Dropout(hp.Float("dropout1", 0.2, 0.5, step=0.1))(y)

    y = base_model.layers[6](y)   # conv2
    y = base_model.layers[7](y)   # conv2_bn
    y = base_model.layers[8](y)   # relu2
    y = base_model.layers[9](y)   # pool2
    y = Dropout(hp.Float("dropout2", 0.2, 0.5, step=0.1))(y)

    y = base_model.layers[10](y)  # conv3
    y = base_model.layers[11](y)  # conv3_bn
    y = base_model.layers[12](y)  # relu3
    y = base_model.layers[13](y)  # pool3
    y = Dropout(hp.Float("dropout3", 0.2, 0.5, step=0.1))(y)

    y = base_model.layers[14](y)  # fc
    output = base_model.layers[15](y)  # softmax

    model = keras.Model(inputs=x, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ======================
# Tuner setup
# ======================
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,             # number of dropout combinations to try
    executions_per_trial=1,    # run each config once
    directory="keras_tuner_dir",
    project_name="blinknet_dropout_tuning"
)

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)



# ======================
# Data generators (with validation split)
# ======================
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    './blinknet_data/train/',
    target_size=(30, 30),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    './blinknet_data/train/',
    target_size=(30, 30),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# ======================
# Run tuner
# ======================
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[reduce_lr]
)


# ======================
# Train best model
# ======================
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"Best dropout rates: "
      f"{best_hps.get('dropout1')}, "
      f"{best_hps.get('dropout2')}, "
      f"{best_hps.get('dropout3')}")

model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[reduce_lr]
)

# Save fine-tuned model
model.save("./blinknet_models/blink_model_finetuned.h5")


# ======================
# Plot Training Curves
# ======================
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()
