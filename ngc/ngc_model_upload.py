
import os
import subprocess

def upload_model_to_ngc(model_path, model_name, version, ngc_repository):
    """
    Uploads a model to NVIDIA GPU Cloud (NGC) with specified metadata.
    Args:
        model_path (str): Path to the model file to be uploaded.
        model_name (str): Name of the model on NGC.
        version (str): Model version for version control.
        ngc_repository (str): NGC repository path (e.g., 'alterlearning/models').
    """
    command = [
        "ngc", "registry", "model", "upload",
        "--source", model_path,
        "--model-name", model_name,
        "--version", version,
        "--repository", ngc_repository
    ]
    subprocess.run(command, check=True)
    print(f"Model {model_name} version {version} uploaded to NGC repository {ngc_repository}.")

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model.pth"
    model_name = "adaptive_vr_model"
    version = "1.0.0"
    ngc_repository = "alterlearning/models"
    upload_model_to_ngc(model_path, model_name, version, ngc_repository)
