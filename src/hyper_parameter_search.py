import tensorflow as tf
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from kerastuner.tuners import RandomSearch
import kerastuner as kt

tf.random.set_seed(123)
# set memory growth for gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

# Define a function to build the CNN model with tunable hyperparameters
def build_cnn_model(hp):
    model = Sequential()
    
    # Define the range of CNN layers to search over (1 to 4 layers)
    num_cnn_layers = hp.Int('num_cnn_layers', min_value=1, max_value=4, step=1)
    
    # Define the range of filters for each CNN layer (16 to 256 filters)
    filters = hp.Choice('filters', values=[16, 32, 64, 128, 256])
    
    # Add the CNN layers
    for i in range(num_cnn_layers):
        model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    # Define the range of dense units for the dense layers (64 to 512 units)
    dense_units = hp.Choice('dense_units', values=[64, 128, 256, 512])
    
    # Add the dense layers
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = RandomSearch(
        build_cnn_model,
        objective='val_accuracy',
        max_trials=1000,  # Number of different models to try
        executions_per_trial=1,  # Number of executions per model
        directory='tuner_directory',
        project_name='cnn_tuner'
    )
start_time = time.time()
# Initialize the tuner within the strategy.scope()
with strategy.scope():

    # Perform the hyperparameter search within the strategy.scope()
    tuner.search(train_dataset, epochs=10, validation_data=validation_dataset)

end_time = time.time()

# Calculate the total training time in hours
total_training_time = (end_time - start_time) / 3600
# Get the best model architecture and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest Hyperparameters:")
print(best_hyperparameters.values)

# Train the best model with the best hyperparameters
best_model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
test_loss, test_accuracy = best_model.evaluate(validation_dataset)
print(f"Test accuracy of the best model: {test_accuracy}")
