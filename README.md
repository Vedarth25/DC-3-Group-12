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

## Setup and Evaluation model 
1. Make sure you have downloaded the DeepFish dataset and have it in the same directory as your project. This project uses data provided by [Tamim662](https://github.com/tamim662/YOLO-Fish). You can access DeepFish dataset we used directly from their project:
- [Data Link](https://drive.google.com/file/d/10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0/view)
2. Download the weights and the cfg files. This project uses the cfg file of yolo-fish-2 available to download from this [link](https://github.com/tamim662/YOLO-Fish/tree/main/models/trained_on_merge) and the merge_yolo-fish-2 weights available to download from this [link](https://drive.google.com/drive/folders/1BmBdxwGCH3IS0kTeDxK2hT8vVvEtd_3o) both provided by [Tamim662](https://github.com/tamim662/YOLO-Fish). Once again make sure you also have them in your working directory.
3. Install all dependencies from requirements.txt and also install DeepSort from PyPI via pip3 install deep-sort-realtime. If you run into any problems with this checkout the official [website](https://pypi.org/project/deep-sort-realtime/) of DeepSort.
4. Open the terminal and run eval.py. If you want to run it with our customized preprocessing pass argument -p, and/or if you also want to run on the test set pass argument -t. (However, using our preprocessing reduces the model performance metrics, so we reccomend against using it).
- **main**: If you are on the main branch you will get the output from the DeepSort which includes the counts of the fishes for all the habitats and also one image from every habitat with the overlayed tracking paths of the fishes on them.
Note! Running this can be computationally heavy and time consuming so please take this into account depending on your machine.
The images with the overlayed paths on them look like this: 
<p align="center">
  <img src="https://github.com/user-attachments/assets/5eac67af-39a3-46da-ae76-0be48c085d38" alt="fish_paths" width="45%" />
  <img src="https://github.com/user-attachments/assets/93e3e313-0900-415b-ad7c-4f50ce47be6c" alt="fish_paths1" width="45%" />
</p>

- **better_metrics**: If you checkout the better_metrics branch you will get the output from the object detection with the YOLOFish. This includes some images with bounding boxes and confidence intervals on them and confusion matrices with the performance metrics, but it does not include tracking or counting. 
Doing this is reccomended if you want to save time and usage of computational resources.

## Acknowledgments
We greatly thank the creators of YOLOFish for making their repository public and the DeepSort team for their reliable real-time tracking solution. 






