
# NVIDIA base image with CUDA for optimized inference
FROM nvcr.io/nvidia/pytorch:22.06-py3

# Install any additional dependencies if needed
RUN apt-get update && apt-get install -y     python3-pip &&     pip3 install tensorrt

# Copy inference script
COPY load_pretrained_ngc.py /app/load_pretrained_ngc.py

# Run inference script by default
CMD ["python3", "/app/load_pretrained_ngc.py"]
