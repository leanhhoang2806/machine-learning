import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Check if GPU is available and enable GPU memory growth to avoid allocation errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the path to the folder containing the image data
data_directory = "/app/data-source/dementia/Data/"

# Specify image dimensions and batch size
image_width, image_height = 224, 224
batch_size = 32

# Use ImageDataGenerator to load and preprocess images
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split the data into training and validation sets
)

# Create a labeled dataset from the images in the subfolders
train_data_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    subset='training',  # Specify that this is the training data
    shuffle=True
)

# Create the validation dataset
validation_data_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Specify that this is the validation data
    shuffle=False  # No need to shuffle the validation data
)

# Load the pre-trained VGG16 model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(image_width, image_height, 3))

# Freeze the pre-trained layers so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add a custom top to the model for the classification task
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(train_data_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
steps_per_epoch = len(train_data_generator)
validation_steps = len(validation_data_generator)
model.fit(
    train_data_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data_generator,
    validation_steps=validation_steps
)

# Save the trained model
# model.save('trained_model.h5')
