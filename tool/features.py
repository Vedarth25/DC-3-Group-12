import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import time


# Load the pre-trained ResNet50 model
class FeatureExtractor:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        # Remove the last fully connected layer to get feature vectors instead of classifications
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()  # Set to evaluation mode

        # Define image transformation pipeline for preprocessing the chips
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the image to PIL format
            transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input size
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for ResNet
        ])

    def extract(self, chip):
        """
        Extract features from the given chip (image crop).
        """
        # Preprocess the image chip
        chip = self.transform(chip).unsqueeze(0)  # Add batch dimension

        # Forward pass through the model to get the feature vector
        with torch.no_grad():
            features = self.model(chip)
        
        # Flatten the feature vector to a 1D tensor
        features = features.view(-1)
        
        return features
    

def plot_fish_paths(fish_paths, habitat_image_path, output_folder, habitat, num_fish_to_plot=None):
    """
    Function to plot the fish paths based on their recorded positions over a habitat image using OpenCV.
    Args:
    - fish_paths: Dictionary where key is fish_id and value is list of (x, y) coordinates for that fish.
    - habitat_image_path: Path to the image of the habitat.
    - output_folder: Folder to save the resulting plot.
    - num_fish_to_plot: Maximum number of fish paths to plot (None to plot all).
    """
    import cv2
    import os
    import numpy as np

    # Load the habitat image
    habitat_image = cv2.imread(habitat_image_path)
    if habitat_image is None:
        raise FileNotFoundError(f"Image not found at {habitat_image_path}")
    
    height, width, _ = habitat_image.shape

    # Limit the number of fish to plot if specified
    fish_ids = list(fish_paths.keys())
    if num_fish_to_plot is not None:
        fish_ids = fish_ids[:num_fish_to_plot]

    # Generate unique colors for each fish path
    num_colors = len(fish_ids)
    colors = np.random.randint(0, 255, size=(num_colors, 3))  # Random RGB colors

    # Draw each fish path on the habitat image
    for idx, fish_id in enumerate(fish_ids):
        path = np.array(fish_paths[fish_id])
        path[:, 0] = path[:, 0]/608*width  # Scale X-coordinate
        path[:, 1] = path[:, 1]/608*height   # Scale Y-coordinate

        # Draw the path line
        for j in range(1, len(path)):
            start_point = (int(path[j-1][0]), int(path[j-1][1]))
            end_point = (int(path[j][0]), int(path[j][1]))
            color = tuple(int(c) for c in colors[idx])  # Convert color to tuple
            cv2.line(habitat_image, start_point, end_point, color, thickness=2)
        
        # Draw a circle at the starting point of each fish path
        start_point = (int(path[0][0]), int(path[0][1]))
        cv2.circle(habitat_image, start_point, radius=5, color=color, thickness=-1)

        # Add fish ID label near the starting point
        label = f'Fish {fish_id}'
        cv2.putText(habitat_image, label, (start_point[0], start_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

    # Add title on the image (optional)
    title_text = f'Fish Paths on Habitat {habitat}'
    cv2.putText(habitat_image, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the resulting image with fish paths
    output_filename = os.path.join(output_folder, f'fish_paths_for_habitat_{habitat}_cv2.png')
    cv2.imwrite(output_filename, habitat_image)
    
    print(f"Fish path plot saved as {output_filename}")
    return habitat_image





  







def overlay_tracks_on_image(image_path, fish_paths, output_folder, habitat, fish_id):
    """
    Function to overlay fish paths on the original image.

    Args:
    - image_path: Path to the habitat image (the first image in the habitat).
    - fish_paths: List of (x, y) coordinates representing the fish path.
    - output_folder: Folder to save the resulting image with the overlay.
    - habitat: The habitat identifier for saving the file.
    - fish_id: The fish ID whose path is being overlayed.
    """
    # Load the original image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from path {image_path}")
        return

    # Convert image from BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    # Convert fish_paths to numpy array for easier processing
    fish_paths = np.array(fish_paths)

    # Overlay the fish path on the image
    plt.plot(fish_paths[:, 0], fish_paths[:, 1], marker='o', label=f'Fish ID: {fish_id}', color='r')

    # Title and labels
    plt.title(f'Fish ID {fish_id} Path on Habitat {habitat}')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')

    # Save the image with the overlaid tracks
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f'overlay_fish_{fish_id}_habitat_{habitat}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved overlay of fish path on image: {save_path}")





def create_tracking_video(frame_folder, fish_paths, output_video_path, frame_size, num_frames, fps=20):
    """
    Create a video to visualize the tracking performance.
    
    Args:
    - frame_folder: Folder containing the individual frames of the video sequence.
    - fish_paths: Dictionary where key is fish_id and value is list of (x, y) coordinates over time.
    - output_video_path: Path where the output video will be saved.
    - frame_size: (width, height) tuple specifying the frame size.
    - num_frames: Total number of frames in the sequence.
    - fps: Frames per second for the output video.
    """
    # OpenCV VideoWriter to create the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Iterate through each frame
    for frame_idx in range(num_frames):
        # Construct the filename for the current frame
        frame_path = os.path.join(frame_folder, f"frame_{frame_idx:04d}.png")  # Adjust file format if necessary
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Frame {frame_idx} could not be loaded.")
            continue

        # Overlay tracking paths for each fish
        for fish_id, path in fish_paths.items():
            if frame_idx < len(path):
                # Get the (x, y) position of the fish at the current frame
                x, y = path[frame_idx]

                # Draw a circle for the fish position
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

                # Put the fish ID near the point
                cv2.putText(frame, f"ID: {fish_id}", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the frame with the overlaid tracking data into the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Tracking video saved to {output_video_path}")
