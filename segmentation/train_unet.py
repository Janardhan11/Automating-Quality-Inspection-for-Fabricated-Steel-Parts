import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2, os
from dataset import load_data

IMG_SIZE = 256

def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    bn = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)

    # Decoder
    u1 = layers.UpSampling2D()(bn)
    concat1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat1)

    u2 = layers.UpSampling2D()(c3)
    concat2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat2)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c4)

    return Model(inputs, outputs)

# Load dataset
X_train, y_train = load_data("../data/train", "../data/masks")
print("Dataset shape:", X_train.shape, y_train.shape)

# Model
model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=8, epochs=20)

# Save model
model.save("../outputs/models/unet_segmentation.h5")
