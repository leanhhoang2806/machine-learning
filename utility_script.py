import os
import shutil

def move_first_100_items(source_folder, destination_folder):
    # Get a list of all items in the source folder
    all_items = os.listdir(source_folder)

    # Ensure the destination folder exists; if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move the first 100 items to the destination folder
    for item in all_items[:100]:
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)
        shutil.move(source_path, destination_path)

if __name__ == "__main__":
    source_folder = "./real-data-source/Non Demented"
    destination_folder = "./data-source/dementia/Data/Non Demented"

    move_first_100_items(source_folder, destination_folder)
