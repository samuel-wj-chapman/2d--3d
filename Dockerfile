FROM nvcr.io/nvidia/pytorch:23.11-py3

#RUN apt-get update --fix-missing && apt-get install -y \
#    nano \
# && rm -rf /var/lib/apt/lists/*

RUN python --version && pip --version
RUN python -m pip install ultralytics
RUN python -m pip install pycocotools


RUN pip uninstall opencv

WORKDIR /app
