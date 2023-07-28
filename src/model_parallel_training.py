# import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import paramiko

# Define the function to connect via SSH and start the TensorFlow worker
def start_worker(hostname, username, password):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=username, password=password)

    # Change the Python script path according to your setup
    python_script_path = '/path/to/your/python_script.py'
    ssh_client.exec_command(f'python {python_script_path}')

# Function to build the neural network model
def build_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Function to set up distributed training using MultiWorkerMirroredStrategy
def setup_distributed_training():
    # Set up the parameter servers' IPs and credentials
    param_server_ips = ['192.168.0.100', '192.168.0.101']
    param_server_usernames = ['hoang', 'hoang2']
    param_server_passwords = ['2910', '2910']

    # Create a list of workers with their respective SSH credentials
    workers = [
        f'{username}:{password}@{ip}'
        for ip, username, password in zip(param_server_ips, param_server_usernames, param_server_passwords)
    ]

    # Set up MultiWorkerMirroredStrategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=tf.distribute.cluster_resolver.SimpleClusterResolver(
            cluster_spec={"worker": workers}
        )
    )

    return strategy

def main():
    # Load and preprocess the MNIST data
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # Set up distributed training strategy
    strategy = setup_distributed_training()

    # Create and compile the model under the strategy scope
    with strategy.scope():
        model = build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using model.fit method
    model.fit(x_train, y_train, epochs=10, batch_size=32)

if __name__ == "__main__":
    main()
