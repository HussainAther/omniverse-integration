
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

# Load a pre-trained model from local storage
def load_model(model_path="model.pt"):
    model = torch.load(model_path)
    model.eval()
    return model

# Sample inference on an image
def run_inference(model, image_path):
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load image and apply preprocessing
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Inference result: {torch.argmax(output)}")

if __name__ == "__main__":
    model = load_model("adaptive_vr_model.pt")
    run_inference(model, "sample_image.jpg")
