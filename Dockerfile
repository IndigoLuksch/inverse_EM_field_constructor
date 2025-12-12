# Use the official Vertex AI GPU image with TensorFlow 2.14 and Python 3.10
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project context to the working directory
COPY . .

# Install the dependencies from requirements.txt
# Note: The base image already contains tensorflow, so it's good to remove it from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for outputs, in case they are not created by the scripts
RUN mkdir -p /app/data /app/models /app/logs /app/results /tfrecords

# Set environment variable to avoid matplotlib display issues
ENV MPLBACKEND=Agg

# Command to run the training script when the container starts
CMD ["python", "run.py"]