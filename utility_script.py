import os
import shutil
import random

current_dir = os.getcwd()
# Define the path to the original data folder
original_data_dir = current_dir + '/data-source/dementia/Data/'

# Define the path to the new data folder with train and test directories
new_data_dir = current_dir + '/test-data-source/dementia/'
train_dir = os.path.join(new_data_dir, 'train')
test_dir = os.path.join(new_data_dir, 'test')

print(f"Original data directory: {original_data_dir}")
print(f"New data directory: {new_data_dir}")
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Create the train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all the classes in the original data folder
classes = os.listdir(original_data_dir)

# Define the ratio of train data to total data
train_data_ratio = 0.8

# Loop through each class and split the images into train and test directories
for class_name in classes:
    class_dir = os.path.join(original_data_dir, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)
    num_train = int(len(images) * train_data_ratio)

    # Move images to the train directory
    for image in images[:num_train]:
        src_path = os.path.join(class_dir, image)
        dst_path = os.path.join(train_dir, class_name, image)
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        shutil.copy(src_path, dst_path)

    # Move images to the test directory
    for image in images[num_train:]:
        src_path = os.path.join(class_dir, image)
        dst_path = os.path.join(test_dir, class_name, image)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        shutil.copy(src_path, dst_path)

print("Data split into train and test folders successfully.")
