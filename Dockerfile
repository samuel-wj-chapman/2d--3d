# Base image with CUDA support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set a default shell
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    wget \
    libgl1-mesa-glx \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip


# Install PyTorch with CUDA support
#RUN pip install torch torchvision torchaudio

# Install other Python dependencies (adjust as needed)
RUN pip install numpy matplotlib opencv-python

# Set the working directory
WORKDIR /workspace


