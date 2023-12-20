# 2d--3d

# 3D Image GAN

This is a personal project aims to convert 2D images into 3D point clouds using a GAN. The project utilizes GPU acceleration with CUDA support.


# Data builder

this requires blender, run the databuilder.py withing blender with your scene already created. The code simply loads an object takes some pictures and saves in the correct format for the dataloader.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)
- [NVIDIA Docker Toolkit](https://github.com/NVIDIA/nvidia-docker)

## Building the Docker Image

To build the Docker image, navigate to the project directory and run:

```bash
docker build -t 3d-image-gan .


docker run --gpus all 3d-image-gan

```
note the docker image may require you to create an account and generate a key with nvidia if you have not already. 





;things to add

colour changes for objects

decimate point clouds - done

