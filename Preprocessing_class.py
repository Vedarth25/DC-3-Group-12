#from google.colab import drive - ONLY if you are using drive 
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 

class ImagePreprocessor:
    def __init__(self, input_folder, target_size=(224, 224)):
        #drive.mount('/content/drive') - ONLY if you are using drive 

        self.input_folder = input_folder
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(('jpg', 'jpeg', 'png'))])

    def apply_clahe_to_color(self, image):
        # Convert the image from BGR to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split LAB image into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L (lightness) channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl_l_channel = clahe.apply(l_channel)

        # Merge the CLAHE enhanced L channel with the A and B channels
        merged_lab = cv2.merge((cl_l_channel, a_channel, b_channel))

        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    def equalize_color_means(self, image):
        # Split the image into its R, G, B channels
        b_channel, g_channel, r_channel = cv2.split(image)

        # Calculate the mean of each channel
        mean_r = np.mean(r_channel)
        mean_g = np.mean(g_channel)
        mean_b = np.mean(b_channel)

        # Calculate the overall mean of all channels
        overall_mean = (mean_r + mean_g + mean_b) / 3

        # Equalize the means by adjusting each channel
        r_channel = np.clip(r_channel + (overall_mean - mean_r), 0, 255).astype(np.uint8)
        g_channel = np.clip(g_channel + (overall_mean - mean_g), 0, 255).astype(np.uint8)
        b_channel = np.clip(b_channel + (overall_mean - mean_b), 0, 255).astype(np.uint8)

        # Merge the adjusted channels back into a single image
        equalized_image = cv2.merge([b_channel, g_channel, r_channel])

        return equalized_image

    def enhance_image(self, image):
        # Apply CLAHE (on color image)
        clahe_image = self.apply_clahe_to_color(image)

        # Apply Color Mean Equalization to the CLAHE result
        final_image = self.equalize_color_means(clahe_image)

        return final_image

    def preprocess_image(self, image):
        # Apply the color correction to the image
        enhanced_img = self.enhance_image(image)

        # Resize image to the target size (if necessary for your model)
        enhanced_img = cv2.resize(enhanced_img, self.target_size)

        # Convert the image from BGR (OpenCV format) to RGB (optional depending on your model)
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        # Expand dimensions to add a batch dimension (if needed)
        #enhanced_img_tensor = np.expand_dims(enhanced_img, axis=0)

        return enhanced_img

    def process_all_images(self, model=None):
        # Loop through all images, apply color correction, and use them in the provided model
        for img_file in self.image_files:
            # Full path to the image
            img_path = os.path.join(self.input_folder, img_file)

            # Load image using OpenCV
            img = cv2.imread(img_path)

            # Preprocess the image
            preprocessed_image = self.preprocess_image(img)

            if model:
                # Use the preprocessed image as input to the provided model (if any)
                predictions = model.predict(preprocessed_image)
                print(f"Processed image {img_file} and got predictions.")
            else:
                print(f"Processed image {img_file} and ready for model input.")

        




        
        #==================================== FOR TESTING- Method to process just one image ===================================
        
    # def process_single_image(self, image_path, model=None):
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         print(f"Error: Could not load image at {image_path}")
    #         return None
        
    #     preprocessed_image = self.preprocess_image(img)
    #     if model:
    #         predictions = model.predict(preprocessed_image)
    #         print(f"Processed single image {os.path.basename(image_path)} and got predictions.")
    #     else:
    #         print(f"Processed single image {os.path.basename(image_path)} and ready for model input.")
        
    #     return preprocessed_image








#UNCOMENT THIS SECTION AND THE def process_single_image IF YOU WANT TO TEST IT ON ONE IMAGE

# # Path to the folder where your images are stored
# input_folder ="path to your image"

# # Initialize the preprocessor with the folder containing the images
# preprocessor = ImagePreprocessor(input_folder)

# # Test with a single image from this folder
# image_path = 'path to your image'

# # Process the single image
# preprocessed_image = preprocessor.process_single_image(image_path)

# # Check if the image was successfully processed
# if preprocessed_image is not None:
#     print("Single image processed successfully.")

#     # Squeeze the batch dimension added earlier (removing the 1st dimension)
#     processed_image = preprocessed_image.squeeze()

#     # Display the processed image using Matplotlib
#     plt.imshow(processed_image)
#     plt.title("Processed Image")
#     plt.axis('off')  # Hide axis for a cleaner display
#     plt.show()



