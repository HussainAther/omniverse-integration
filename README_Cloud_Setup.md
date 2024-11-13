# Alter Learning Cloud Setup with NVIDIA Omniverse and Oracle Cloud Infrastructure (OCI)

This repository provides a comprehensive guide for setting up and deploying NVIDIA Omniverse on Oracle Cloud Infrastructure (OCI), leveraging NVIDIA Nucleus, Omniverse Kit, and Omniverse Create for a collaborative VR/AR and AI-driven educational platform.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
  - [1. Set Up an OCI Account and Configure Environment](#1-set-up-an-oci-account-and-configure-environment)
  - [2. Install NVIDIA Drivers and CUDA on OCI](#2-install-nvidia-drivers-and-cuda-on-oci)
  - [3. Deploy Omniverse Nucleus on OCI](#3-deploy-omniverse-nucleus-on-oci)
  - [4. Set Up Omniverse Kit and Create](#4-set-up-omniverse-kit-and-create)
  - [5. Optimize OCI Instances for Performance](#5-optimize-oci-instances-for-performance)
  - [6. Monitoring and Analytics](#6-monitoring-and-analytics)
- [Additional Resources](#additional-resources)

---

## Overview

The goal of this setup is to establish a scalable, collaborative cloud environment for Alter Learning’s VR/AR and AI applications, using NVIDIA Omniverse on Oracle Cloud Infrastructure (OCI). This guide covers configuring NVIDIA Nucleus for real-time collaboration, using Omniverse Kit and Create for VR/AR development, and setting up monitoring tools for system optimization.

## Prerequisites

1. **Oracle Cloud Account**: Sign up at [oracle.com/cloud](https://oracle.com/cloud) (consider the free trial for testing).
2. **NVIDIA Omniverse Software**: Install **Nucleus**, **Omniverse Kit**, and **Omniverse Create**.
3. **Access to NVIDIA Developer Tools**: If required, sign up for a developer account at [NVIDIA Developer](https://developer.nvidia.com/).

## Step-by-Step Setup

### 1. Set Up an OCI Account and Configure Environment

1. **Create an OCI Account**: Sign up for Oracle Cloud and log in to the console.
2. **Select a GPU-Enabled Compute Instance**:
   - Choose an NVIDIA GPU instance (e.g., **BM.GPU4.8** for V100 GPUs or **BM.GPU.GPU.A100** for A100 GPUs).
   - Configure the instance with at least 16GB of RAM (32GB recommended for heavy workloads) and open network access.
3. **Configure Networking**:
   - Set up a **Virtual Cloud Network (VCN)** in OCI.
   - Open necessary ports (e.g., 3009 for Nucleus) for secure remote access.

### 2. Install NVIDIA Drivers and CUDA on OCI

1. **Install NVIDIA Drivers**:
   - SSH into your OCI instance.
   - Run the following commands to install NVIDIA drivers (adjust driver version as needed):
     ```bash
     sudo apt update
     sudo apt install -y nvidia-driver-450
     ```
2. **Install CUDA Toolkit**:
   - Install CUDA for AI tasks and GPU acceleration:
     ```bash
     sudo apt update
     sudo apt install -y nvidia-cuda-toolkit
     ```

### 3. Deploy Omniverse Nucleus on OCI

1. **Download and Install Nucleus Server**:
   - Download the **Nucleus Server** package from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/).
   - Install it on your OCI instance:
     ```bash
     wget <URL to Nucleus download>
     sudo dpkg -i nucleus-server-package.deb
     ```
2. **Configure Nucleus for Remote Access**:
   - Open port **3009** to allow access to Nucleus.
   - Set up permissions in the Nucleus Admin UI to manage access control for team members.

### 4. Set Up Omniverse Kit and Create

1. **Install Omniverse Kit on OCI**:
   - Install Omniverse Kit on your cloud instance (or a separate instance if desired).
   - Connect Kit to Nucleus by entering the Nucleus server IP in **File > Connect to Nucleus**.
2. **Set Up Omniverse Create**:
   - Install Omniverse Create on local devices or cloud instances for collaborative development.
   - Connect Omniverse Create to Nucleus to access shared assets and enable real-time teamwork.

### 5. Optimize OCI Instances for Performance

1. **Enable Auto-Scaling**:
   - Configure auto-scaling in OCI to increase or decrease instance count based on demand.
2. **Set Up Load Balancing**:
   - Use OCI Load Balancer to distribute traffic across multiple instances if needed, ensuring stable performance and scalability.

### 6. Monitoring and Analytics

1. **Set Up OCI Monitoring**:
   - Use OCI’s built-in monitoring tools to track GPU, CPU, and memory usage.
2. **Integrate Prometheus and Grafana for Custom Dashboards**:
   - For detailed analytics, install Prometheus to collect metrics from Nucleus and Omniverse Kit.
   - Visualize metrics in Grafana for real-time monitoring and alerting.

## Additional Resources

- **Omniverse Documentation**: [Omniverse Developer Guide](https://docs.omniverse.nvidia.com/dev-guide/latest/overview.html)
- **Oracle Cloud Documentation**: [OCI Documentation](https://docs.oracle.com/en/cloud/)
- **Omniverse GitHub Repository**: [NVIDIA Omniverse GitHub](https://github.com/NVIDIA-Omniverse)


