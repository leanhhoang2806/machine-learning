import os
from PIL import Image
from ..utilities.Immutables import ImmutableArray

def load_and_resize_image(image_path, image_shape):
    image = Image.open(image_path)
    image = image.resize(image_shape)
    image_array = list(image.getdata())
    width, height = image.size
    # Convert RGB tuples to lists of integers
    image_array = [list(pixel) for pixel in image_array]
    return ImmutableArray([image_array[i:i+width] for i in range(0, len(image_array), width)])

def extract_images_as_batches(folder_path, batch_size, image_shape):
    file_list = os.listdir(folder_path)
    num_files = len(file_list)
    num_batches = (num_files + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_images = []

        for idx in range(batch_size):
            file_idx = batch_idx * batch_size + idx
            if file_idx >= num_files:
                break

            image_path = os.path.join(folder_path, file_list[file_idx])
            image = load_and_resize_image(image_path, image_shape)
            batch_images.append(image)

        yield batch_images