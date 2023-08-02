import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
import time
from sklearn.model_selection import KFold

# Check if GPU is available and enable GPU memory growth to avoid allocation errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the path to the folder containing the image data
data_directory = "/app/data-source/dementia/Data/"

# Specify image dimensions and batch size
image_width, image_height = 112, 112
batch_size = 5

# Use ImageDataGenerator to load and preprocess images
datagen = ImageDataGenerator(
    rescale=1.0/255.0)  # Normalize pixel values to [0, 1]

# Create a labeled dataset from the images in the subfolders
data_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for one-hot encoded labels
    shuffle=True
)

# Define a list of neural network models to evaluate
models = [
    # ("Dense Neural Network", Sequential([
    #     Flatten(input_shape=(image_width, image_height, 3)),
    #     Dense(256, activation='relu'),
    #     Dense(128, activation='relu'),
    #     Dense(data_generator.num_classes, activation='softmax')
    # ])),
    # ("VGG16 Pre-trained Model", Sequential([
    #     VGG16(include_top=False, weights='imagenet', input_shape=(image_width, image_height, 3)),
    #     GlobalAveragePooling2D(),
    #     Dense(data_generator.num_classes, activation='softmax')
    # ])),
    # ("ResNet50 Pre-trained Model", Sequential([
    #     ResNet50(include_top=False, weights='imagenet', input_shape=(image_width, image_height, 3)),
    #     GlobalAveragePooling2D(),
    #     Dense(data_generator.num_classes, activation='softmax')
    # ])),
    # ("Simple CNN Model", Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(data_generator.num_classes, activation='softmax')
    # ])),
    ("Complex CNN Model", Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(data_generator.num_classes, activation='softmax')
    ])),
    # Add more neural network architectures as needed
]

# Record the start time of the model selection process
start_time = time.time()

# Perform cross-validation and evaluate models
model_accuracies = []
n_splits = 5
for name, model in models:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kfold.split(data_generator):
        train_data = tf.data.Dataset.from_generator(
            lambda: ((data_generator[i][0], data_generator[i][1])for i in train_index),
            output_signature=(
                tf.TensorSpec(shape=(None, image_width, image_height, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, data_generator.num_classes), dtype=tf.float32)
            )
        ).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

        val_data = tf.data.Dataset.from_generator(
            lambda: ((data_generator[i][0], data_generator[i][1]) for i in val_index),
            output_signature=(
                tf.TensorSpec(shape=(None, image_width, image_height, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, data_generator.num_classes), dtype=tf.float32)
            )
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        # Calculate the number of steps for each epoch
        steps_per_epoch = len(train_index) // batch_size
        validation_steps = len(val_index) // batch_size
        print(f"Training {name} with {steps_per_epoch} steps per epoch and {validation_steps} validation steps per epoch")
        model.fit(train_data, epochs=10, steps_per_epoch=steps_per_epoch,
                  validation_data=val_data, validation_steps=validation_steps, verbose=0)
        
        _, accuracy = model.evaluate(val_data)
        scores.append(accuracy)

    mean_accuracy = np.mean(scores)
    model_accuracies.append((name, mean_accuracy))
    print(f"{name} - Cross-validation Accuracy: {mean_accuracy:.2f} (+/- {np.std(scores):.2f}) - Train time {time.time() - start_time:.2f} seconds")

# Choose the best performing model based on cross-validation results
best_model_name, best_model_accuracy = max(model_accuracies, key=lambda x: x[1])
print("\nBest Model:", best_model_name)
print("Best Model Accuracy:", best_model_accuracy)

# Record the end time of the model selection process
end_time = time.time()

# Calculate and print the duration of the model selection process
duration = end_time - start_time
print("Model selection duration:", duration // 3600 , "hours")

# Train the best model on the full dataset and save the model
best_model = next(model for name, model in models if name == best_model_name)
best_model.fit(data_generator, epochs=10)  