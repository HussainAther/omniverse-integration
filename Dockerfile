
# Base image with CUDA support
FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Install Python and essential packages
RUN apt-get update && apt-get install -y     python3     python3-pip

# Install PyTorch with CUDA and TensorRT support
RUN pip3 install torch torchvision     && pip3 install pycuda tensorrt

# Set working directory
WORKDIR /app

# Copy training and inference scripts
COPY train_model_cuda.py /app/train_model_cuda.py
COPY inference_with_tensorrt.py /app/inference_with_tensorrt.py

# Set default command to run the training script
CMD ["python3", "train_model_cuda.py"]
