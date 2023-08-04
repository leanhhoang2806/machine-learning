import tensorflow as tf
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
## Working model version 0.1

# Define the number of workers
num_workers = 2

# Initialize the distributed strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Define the data directories
current_dir = os.getcwd()
# Define the path to the original data folder
original_data_dir = current_dir + '/data-source/dementia/Data/'

# Define the path to the new data folder with train and test directories
train_data_dir = current_dir + '/test-data-source/dementia/train'
validation_data_dir = current_dir + '/test-data-source/dementia/test'

# Define some constants for the training
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_workers
IMG_HEIGHT, IMG_WIDTH = 56, 56

# Prepare the data for training
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# AUTOTUNE is used to automatically tune the dataset prefetching based on available resources.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# List to store validation accuracies for each model
validation_accuracies = []

# Define the model inside the strategy scope
with strategy.scope():
    # Define a list of model architectures with names to search for the best model
    model_architectures = [
        {
            'name': 'Dense Model',
            'model': tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
        },
        {
            'name': 'VGG16 Pre-trained Model',
            'model': Sequential([
                        VGG16(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        GlobalAveragePooling2D(),
                        Dense(4, activation='softmax')
                    ])
        },
        {
            'name': 'ResNet50 Pre-trained Model',
            'model': Sequential([
                        ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        GlobalAveragePooling2D(),
                        Dense(4, activation='softmax')
                    ])
        },
        {
            'name': 'Convo CNN Model',
            'model': Sequential([
                        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        MaxPooling2D((2, 2)),
                        Conv2D(128, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Conv2D(256, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(256, activation='relu'),
                        Dense(4, activation='softmax')
                    ]),

        }
    ]
    for model_info in model_architectures:
        model_name = model_info['name']
        model = model_info['model']
        print(f"Training {model_name} with architecture: {model}")

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])
        # Define the number of epochs
        num_epochs = 10

        # Calculate the time taken for data preprocessing and training
        start_time = time.time()
        # Start distributed training
        model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert elapsed time to hours
        hours = elapsed_time / 3600
        print(f"Training completed in {hours:.2f} hours.")
           # Evaluate the model on the validation dataset
        _, eval_acc = model.evaluate(validation_dataset)
        validation_accuracies.append((model_name, eval_acc))
        print(f"{model_name} validation accuracy: {eval_acc}")

# Function to count the number of image files in a directory
def count_images(directory):
    num_images = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                num_images += 1
    return num_images


# Count the number of images in the train directory
num_train_images = count_images(train_data_dir)
print(f"Number of images in the train directory: {num_train_images}")


# Find the best model with the highest validation accuracy
best_model_info = max(validation_accuracies, key=lambda x: x[1])
best_model_name, best_model_val_acc = best_model_info

print(f"\nBest model architecture: {best_model_name}")
print(f"Best model validation accuracy: {best_model_val_acc}")
