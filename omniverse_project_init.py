
import os

# Set up directories for Omniverse project
def initialize_project():
    project_directories = [
        'assets', 'assets/textures', 'scripts', 'scenes', 'notebooks'
    ]

    for directory in project_directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Example setup message
    print("Omniverse project initialized with required directories and structure.")

if __name__ == "__main__":
    initialize_project()
