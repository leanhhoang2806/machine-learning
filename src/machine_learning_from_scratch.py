import tensorflow as tf
import json
import os

BUFFER_SIZE = 10000
BATCH_SIZE = 4

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

def train_task(index):
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["192.168.1.101:2222","192.168.1.100:2222"],
        },
        'task': {'type': 'worker', 'index': index},
    })
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        multi_worker_model = build_and_compile_cnn_model()

    multi_worker_model.fit(x_train, y_train, epochs=3)

# runs on ip1
train_task(0)
# runs on ip2
# train_task(1)