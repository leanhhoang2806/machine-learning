# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu as Builder

# Install Conda
# Install required packages for downloading Miniconda
# Install PyCUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

FROM tensorflow/tensorflow:latest-gpu as PipInstaller

# Copy system-level packages from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container's working directory
COPY . .
# COPY requirements.txt .
# COPY data-source/ .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Set the entry point to run main.py when the container starts
# ENTRYPOINT ["python", "-m", "src.distributed_training"]
# ENTRYPOINT [ "python", "-m", "src.single_machine_training" ]
ENTRYPOINT ["python", "-m", "src.tf_record_analysis"]
# ENTRYPOINT [ "ls", "-la", "/root/.ssh/" ]
