# DC-3-Group-12
Project is done for Data Challenge 3 (JBG060) course with the support of Technical University of Eindhoven, University of Tilburg, FruitPunchAI and ReefSupport.
## Overview

This project combines the YOLO (You Only Look Once) object detection algorithm with the **DeepSORT** tracking algorithm to detect and track fish in underwater images and videos with the intention to count unique instances. The YOLOFish model and toolset, originally developed by [Tamim662](https://github.com/tamim662/YOLO-Fish), has been integrated with DeepSORT to assign unique IDs to each detected fish, allowing for tracking across multiple frames per habitat.

### Key Features
- **Fish Detection**: Detects multiple fish species using YOLOv4.
- **Object Tracking**: Tracks detected fish across frames using DeepSORT.
- **Unique Fish IDs**: Each fish is assigned a unique ID for tracking.
- **Optional Customizable Preprocessing**: Image preprocessing pipeline for resizing and color normalization.
  
## Model Architecture

1. **YOLOv4**: A single-stage object detector that uses a CNN to predict bounding boxes and class probabilities directly from full images. It is fast and effective for detecting objects in real-time.
2. **DeepSORT**: A robust tracking algorithm that uses Kalman filters and a deep learning-based appearance descriptor to track objects over time and assign consistent IDs.