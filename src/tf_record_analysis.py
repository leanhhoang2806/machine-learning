import tensorflow as tf

# Path to the directory containing uncompressed scenario files
data_dir = '/app/tf_records'

# List all the scenario files in the directory
scenario_files = tf.data.Dataset.list_files(data_dir)

# Function to parse the scenario data
def parse_scenario(example):
    feature_description = {
        'scenario/id': tf.io.FixedLenFeature([], tf.string),
        'scenario/context/sequence_id': tf.io.FixedLenFeature([], tf.int64),
        # Add more features you want to parse here
    }
    return tf.io.parse_single_example(example, feature_description)

# Create a dataset of parsed scenario data
parsed_scenario_data = scenario_files.interleave(
    lambda filename: tf.data.TFRecordDataset(filename),
    cycle_length=tf.data.experimental.AUTOTUNE,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).map(parse_scenario)

# Iterate through the parsed scenario data
for scenario in parsed_scenario_data.take(5):  # Take 5 examples for demonstration
    print(scenario['scenario/id'])
    print(scenario['scenario/context/sequence_id'])
    # Print other parsed features

# (Optional) Prefetch and batch the dataset for training
batch_size = 32
prefetched_scenario_data = parsed_scenario_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
batched_scenario_data = prefetched_scenario_data.batch(batch_size)

# (Optional) Iterate through the batched scenario data
for batch in batched_scenario_data.take(3):  # Take 3 batches for demonstration
    print(batch['scenario/id'])
    print(batch['scenario/context/sequence_id'])
    # Print other batched features
