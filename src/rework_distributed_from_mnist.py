import tensorflow as tf
import os

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
IMG_HEIGHT, IMG_WIDTH = 28, 28

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

# Define the model inside the strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

# Define the number of epochs
num_epochs = 10

# Start distributed training
model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)

# No test dataset in this case, as we have used a separate validation dataset for evaluation.
