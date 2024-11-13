
# Use NVIDIA base image with CUDA support for Omniverse compatibility
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y     python3     python3-pip     wget     libgl1-mesa-glx

# Install NVIDIA Omniverse and other necessary packages
RUN pip3 install --upgrade pip &&     pip3 install jupyterlab

# Expose ports for Jupyter Lab and Omniverse access
EXPOSE 8888 3009

# Set up default entrypoint for interactive Omniverse and JupyterLab usage
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
