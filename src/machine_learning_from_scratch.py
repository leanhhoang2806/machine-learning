import os
import tensorflow as tf
import json


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def train_model():

    cluster_spec = {
        "worker": [
            "{}:2222".format('192.168.1.100'),
            "{}:2222".format('192.168.1.101'),
            ]
    }

    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_spec,
        "task": {"type": "worker", "index": 0}
    })

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = build_model()
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=64)

if __name__ == "__main__":
    train_model()
