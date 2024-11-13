# OpenUSD Collaboration and Version Control Pipeline

This folder contains scripts, examples, and documentation for setting up a collaborative OpenUSD pipeline using NVIDIA Omniverse and Oracle Cloud Infrastructure (OCI). This pipeline enables Alter Learning’s team to work seamlessly on 3D assets, ensuring non-destructive editing, real-time synchronization, and version control.

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
  - [1. Initial Scene Setup](#1-initial-scene-setup)
  - [2. Non-Destructive Layered Editing](#2-non-destructive-layered-editing)
  - [3. Real-Time Collaboration with Nucleus](#3-real-time-collaboration-with-nucleus)
- [Version Control with Omniverse Nucleus](#version-control-with-omniverse-nucleus)
- [Additional Resources](#additional-resources)

---

## Overview

This pipeline leverages **OpenUSD** to create a unified, structured approach to data exchange, version control, and collaborative 3D asset management. With Omniverse Nucleus, team members can collaborate on shared assets in real time, utilizing non-destructive workflows and versioning for maximum flexibility.

### Key Features

- **Unified Data Format**: Use OpenUSD’s standardized format to share 3D assets across platforms.
- **Non-Destructive Workflow**: Layer-based editing preserves original assets, allowing multiple contributors to work concurrently.
- **Real-Time Synchronization**: Omniverse Nucleus provides real-time updates across applications, enabling seamless teamwork.

---

## Folder Structure

Here’s an overview of the files and folders within `openusd_pipeline`:

- **`scene_setup.py`**: Initializes the main scene with geometry, shading, and lighting.
- **`layered_editing_example.py`**: Demonstrates non-destructive edits with USD layers.
- **`nucleus_collaboration_example.py`**: Shows how to connect and save data to Omniverse Nucleus.
- **`version_control.md`**: Documentation on using Omniverse Nucleus for version control, with guidance on locking, notifications, and conflict resolution.

---

## Setup Instructions

### 1. Initial Scene Setup

The `scene_setup.py` script creates a basic USD scene file, defining geometry, materials, and lighting. This is the starting point for any new asset or environment you want to collaboratively edit.

1. Run the following command in your terminal:
   ```bash
   python scene_setup.py
   ```
2. This generates a file named `my_scene.usda` with the initial 3D scene structure.

### 2. Non-Destructive Layered Editing

The `layered_editing_example.py` script shows how to create layers for non-destructive edits. Team members can each work on different layers to apply changes without modifying the base scene.

1. Open the base scene using the script:
   ```bash
   python layered_editing_example.py
   ```
2. This script creates a new layer (`my_scene_edit.usda`) and saves changes, demonstrating how edits are kept separate from the base file.

### 3. Real-Time Collaboration with Nucleus

The `nucleus_collaboration_example.py` script enables real-time collaboration on the Nucleus server. It connects to a shared asset hosted on Omniverse Nucleus, allowing team members to make and view changes simultaneously.

1. Run the following command to connect to Nucleus and make changes:
   ```bash
   python nucleus_collaboration_example.py
   ```
2. Ensure you have access to the Nucleus server and permissions set to edit the shared asset.

---

## Version Control with Omniverse Nucleus

For advanced version control, refer to the `version_control.md` file. This document includes guidelines on:

- **Asset Locking**: Lock assets to prevent concurrent edits.
- **Notifications**: Enable notifications for updates or changes made by other team members.
- **Versioning**: Create tagged versions of assets to track changes over time and revert if needed.

### Usage Tips

- Use meaningful names for layers and version tags to track changes effectively.
- Coordinate with team members when making significant updates to avoid conflicts.
- Regularly check for Nucleus notifications to stay informed of updates.

---

## Additional Resources

- [NVIDIA Omniverse Documentation](https://docs.omniverse.nvidia.com)
- [OpenUSD Documentation](https://graphics.pixar.com/usd/docs/index.html)
- [Oracle Cloud Infrastructure Documentation](https://docs.oracle.com/en/cloud/)

