# Use a lighter base image
FROM python:3.10-slim

# Install system dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch (CPU version for faster build if GPU is not strictly required for smoke test, 
# but the task implies deep learning, so we\'ll stick to a standard installation 
# but optimized for the current environment).
# To make it truly fast and light, we install torch first.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (respecting .dockerignore)
COPY . .
