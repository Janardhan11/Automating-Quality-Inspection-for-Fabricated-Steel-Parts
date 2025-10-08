import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths
train_dir = "../data/train"

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, subset="training"
)
val_gen = datagen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, subset="validation"
)

# Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze backbone

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(2, activation="softmax")  # Defect / No Defect
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=15)

# Save model
model.save("../outputs/models/resnet50_classifier.h5")
