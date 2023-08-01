import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
import time
from sklearn.model_selection import KFold


# Define a function to read and preprocess images from their file paths
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (image_width, image_height))
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, depth=data_generator.num_classes)
    return image, label
# Check if GPU is available and enable GPU memory growth to avoid allocation errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the path to the folder containing the image data
data_directory = "/app/data-source/dementia/Data/"

# Specify image dimensions and batch size
image_width, image_height = 112, 112
batch_size = 5
# Record the start time of the model selection process
start_time = time.time()

# Perform cross-validation and evaluate models
model_accuracies = []
n_splits = 5
strategy = tf.distribute.MultiWorkerMirroredStrategy() # strategy scope always must be the line after the strategy initialization
with strategy.scope():
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

    for name, model in models:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_index, val_index in kfold.split(data_generator.filenames):  # Modified here
            train_filenames = [data_generator.filenames[i] for i in train_index]
            train_classes = [data_generator.classes[i] for i in train_index]
            train_classes_onehot = tf.one_hot(train_classes, depth=data_generator.num_classes)  # Convert class labels to one-hot encoded format
            train_data = tf.data.Dataset.from_tensor_slices((train_filenames, train_classes_onehot))
            train_data = train_data.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_data = train_data.cache()  # Cache the preprocessed data
            train_data = train_data.batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            val_filenames = [data_generator.filenames[i] for i in val_index]
            val_classes = [data_generator.classes[i] for i in val_index]
            val_classes_onehot = tf.one_hot(val_classes, depth=data_generator.num_classes)  # Convert class labels to one-hot encoded format
            val_data = tf.data.Dataset.from_tensor_slices((val_filenames, val_classes_onehot))
            val_data = val_data.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_data = val_data.cache()  # Cache the preprocessed data
            val_data = val_data.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
duration = (end_time - start_time) / 3600.0
print(f"\nModel selection completed in {duration:.2f} hours")

# Train the best model on the full dataset and save the model
best_model = next(model for name, model in models if name == best_model_name)
best_model.fit(data_generator, epochs=10)  
