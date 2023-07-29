import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import paramiko

# Function to build the neural network model
def build_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Function to start the TensorFlow worker on a remote machine
def start_worker(hostname, username, private_key_path, passphrase):
    ssh_client = paramiko.SSHClient()
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path, password=passphrase)

    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=username, pkey=private_key)

def main():
    # Load and preprocess the MNIST data
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # Set up the IP addresses
    master_ip = '192.168.1.100'
    worker_ip = '192.168.1.101'

    # Start TensorFlow worker on the worker node (assuming passwordless SSH is set up)
    start_worker(worker_ip, 'hoang2', private_key_path='/root/.ssh/master_node_id_rsa', passphrase='2910')

    # Define the cluster_spec with master and worker tasks
    cluster_spec = tf.train.ClusterSpec({
        'master': [f'{master_ip}:2222'],
        'worker': [f'{worker_ip}:2222']
    })

    # Create the server to coordinate the tasks
    server = tf.distribute.Server(cluster_spec, job_name='master', task_index=0)

    # Create the strategy based on the server
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Create and compile the model under the strategy scope
    with strategy.scope():
        model = build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using model.fit method
    model.fit(x_train, y_train, epochs=10, batch_size=32)

if __name__ == "__main__":
    main()
