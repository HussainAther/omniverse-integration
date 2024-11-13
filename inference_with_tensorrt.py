
import torch
import tensorrt as trt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define TensorRT logger and builder
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_model_and_optimize():
    # Load the pre-trained PyTorch model
    model = torch.load("simple_nn_model.pth")
    model.eval()

    # Optimize model with TensorRT (example approach, simplified for demo)
    with torch.no_grad():
        print("Converting PyTorch model to TensorRT optimized model.")
        # Note: Full conversion steps would require additional TensorRT model export and setup
        # Simulated optimization here for demonstration
        optimized_model = model  # Replace with actual TensorRT conversion steps as needed
    return optimized_model

def run_inference(model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # Run inference
    for images, _ in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        print(f"Predicted label: {predicted.item()}")
        break  # Run inference on one example for demo purposes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimized_model = load_model_and_optimize()
    run_inference(optimized_model, device)
