
# Omniverse Integration

This repository contains all necessary files and configurations for integrating NVIDIA Omniverse with Alter Learning's immersive 3D and collaborative learning modules. By leveraging Omniverse, we aim to create realistic simulations and interactive environments, enabling students to engage in virtual labs and simulations.

## Overview

The Omniverse Integration project provides a platform for collaborative 3D content creation, simulation, and real-time interaction within virtual learning environments. Key features include support for physics-based simulations using Omniverse PhysX, containerized deployment for consistency, and Kubernetes support for scalability.

## Repository Structure

```
omniverse-integration/
├── omniverse_project_init.py       # Script for setting up and initializing an Omniverse project
├── physx_simulation.py             # Script for physics-based simulations using Omniverse PhysX
├── Dockerfile                      # Docker configuration for setting up Omniverse in a container
├── kubernetes/
│   ├── omniverse_deployment.yaml   # Kubernetes deployment configuration for Omniverse
│   └── omniverse_service.yaml      # Service configuration to expose Omniverse for collaboration
├── assets/
│   ├── lab_environment.usd         # Sample 3D lab environment in USD format
│   └── textures/                   # Textures and other large asset files for 3D environments
├── docs/
│   └── README.md                   # Documentation for setup, usage, and troubleshooting
└── notebooks/                      # Jupyter notebooks for running collaborative simulations
```

## Getting Started

### Prerequisites

- **NVIDIA Omniverse** installed and configured
- **Docker** for containerized environments
- **Kubernetes** for scalable deployment
- **Python 3.8+** for running project scripts
- **NVIDIA GPU** on the deployment machine for optimal performance

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/omniverse-integration.git
   cd omniverse-integration
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t omniverse-env .
   docker run --gpus all -p 8888:8888 omniverse-env
   ```

3. Deploy on Kubernetes:
   - Apply the deployment configurations in the `kubernetes/` folder:
     ```bash
     kubectl apply -f kubernetes/omniverse_deployment.yaml
     kubectl apply -f kubernetes/omniverse_service.yaml
     ```

4. Initialize the Omniverse project:
   ```bash
   python omniverse_project_init.py
   ```

## Usage

- **Run Collaborative Simulations**: Use the provided Jupyter notebooks to launch collaborative simulations.
- **3D Physics Simulations**: Modify `physx_simulation.py` to create custom experiments for virtual labs.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature-name'`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

By following this structure, we ensure the Omniverse Integration repository remains organized, easily navigable, and set up for scalable collaborative learning experiences.
