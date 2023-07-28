import numpy as np

shape = (5, 64, 28, 28)
dtype = np.float32

# Calculate the number of elements in the tensor
num_elements = np.prod(shape)

# Calculate the memory needed in bytes
bytes_per_element = np.dtype(dtype).itemsize
memory_needed_bytes = num_elements * bytes_per_element

# Convert bytes to other units for readability
memory_needed_kb = memory_needed_bytes / 1024
memory_needed_mb = memory_needed_kb / 1024
memory_needed_gb = memory_needed_mb / 1024

print(f"Memory needed: {memory_needed_bytes} bytes")
print(f"Memory needed: {memory_needed_kb:.2f} KB")
print(f"Memory needed: {memory_needed_mb:.2f} MB")
print(f"Memory needed: {memory_needed_gb:.2f} GB")
