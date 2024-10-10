import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tool.darknet2pytorch import Darknet
from tool.utils import *
from tool.torch_utils import *
import torch
from tqdm import tqdm  # Import tqdm for progress bar
import argparse
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

"""hyper parameters"""
use_cuda = True

def detect_fish_in_image(imgfile, m, pre=False):
    """
    Function to detect fish in a given image using YOLO model.
    Returns the number of fish detected.
    """


    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    if pre: 
        #TODO implement pre processing
        print("pre processing not implemented")
        

    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    
    # Assuming that each box is a fish detection
    num_fish_detected = len(boxes[0])  # Each box is a fish detection

    return num_fish_detected

def evaluate_model(csv_file, image_folder, output_txt, m, pre=False):
    """
    Function to evaluate the YOLO model against images from the CSV file.
    Records the mismatches and saves the result in a text file.
    Also tracks over-detection, under-detection for confusion matrix.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    total_images = len(data)
    correct_detections = 0
    under_detections = 0
    over_detections = 0

    mismatch_ids = []

    ground_truth_list = []
    predicted_list = []

    # Loop over each row in the CSV
    for index, row in tqdm(data.iterrows(), total=total_images, desc="Evaluating images", unit="img"):
        img_id = row['ID']
        ground_truth_count = row['counts']

        # Build the image path
        img_path = os.path.join(image_folder, img_id + '.jpg')

        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found!")
            continue

        # Detect fish in the image
        detected_count = detect_fish_in_image(img_path, m, pre)

        ground_truth_list.append(ground_truth_count)
        predicted_list.append(detected_count)

        # Compare with ground truth
        if detected_count == ground_truth_count:
            correct_detections += 1
        elif detected_count < ground_truth_count:
            under_detections += 1
            mismatch_ids.append(img_id)
        else:
            over_detections += 1
            mismatch_ids.append(img_id)

    # Save mismatches to a txt file
    with open(output_txt, 'w') as f:
        for img_id in mismatch_ids:
            f.write(f"{img_id}\n")

    # Return statistics
    return total_images, correct_detections, under_detections, over_detections, ground_truth_list, predicted_list

def evaluate_from_test_folder(test_folder, m, pre=False):
    """
    Evaluate the YOLO model on a test folder of images with associated text files.
    """
    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    total_images = len(image_files)

    all_ground_truths = []
    all_predictions = []

    for img_file in tqdm(image_files, total=total_images, desc="Evaluating images from test folder", unit="img"):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(test_folder, img_file)
        txt_path = os.path.join(test_folder, img_id + '.txt')

        if not os.path.exists(txt_path):
            print(f"Ground truth file {txt_path} not found!")
            continue

        # Load ground truth from txt file (format: class_id, center_x, center_y, width, height)
        with open(txt_path, 'r') as f:
            ground_truth_count = len(f.readlines())  # Each line represents one object

        detected_count, _ = detect_fish_in_image(img_path, m, pre)
        all_ground_truths.append(ground_truth_count)
        all_predictions.append(detected_count)

    return all_ground_truths, all_predictions

def create_infographic(total_images, correct_detections, under_detections, over_detections):
    """
    Function to create and save an infographic (pie chart) showing the model's performance.
    """
    labels = 'Correct', 'Under-detections', 'Over-detections'
    sizes = [correct_detections, under_detections, over_detections]
    colors = ['green', 'orange', 'red']
    explode = (0.1, 0, 0)  # explode 1st slice

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('YOLO Model Performance: Correct, Under, Over Detections')
    
    plt.savefig('model_performance_infographic.png')
    plt.show()

def create_confusion_matrix(ground_truth_list, predicted_list):
    """
    Function to create and display/save a confusion matrix showing actual vs predicted fish counts.
    """
    plt.figure(figsize=(10, 8))
    data = {'Ground Truth': ground_truth_list, 'Predicted': predicted_list}
    df = pd.DataFrame(data, columns=['Ground Truth', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Ground Truth'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'], dropna=False)
    
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix: Actual vs Predicted Fish Counts')
    plt.savefig('confusion_matrix.png')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-p', '--preprocess', action='store_true',
                        help='enable pre-processing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='make it test agains test folder')
    args = parser.parse_args()
    return args

def calculate_metrics(ground_truths, predictions):
    """
    Calculate precision, recall, F1-score, and average precision.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='weighted', zero_division=0)
    average_precision = average_precision_score(ground_truths, predictions)

    return precision * 100, recall * 100, f1 * 100, average_precision * 100

if __name__ == '__main__':
    # Configuration files
    cfgfile = 'yolo-fish-2.cfg'
    weightfile = 'merge_yolo-fish-2.weights'
    csv_file = 'data/Localization.csv'
    image_folder = 'data/images'
    output_txt = 'mismatch_ids.txt'
    args = get_args()
    if args.p:
        pre = True

    m = Darknet(cfgfile)
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    if args.t:
        ground_truths, predictions = evaluate_from_test_folder('test', m, pre)
        create_confusion_matrix(all_ground_truths, all_predictions)
        exit(0)
    else:
        total_images, correct_detections, under_detections, over_detections, ground_truth_list, predicted_list = evaluate_model(csv_file, image_folder, output_txt, m, pre)
    
    precision, recall, f1_score, ap = calculate_metrics(ground_truths, predictions)

    # Create the infographic
    create_infographic(total_images, correct_detections, under_detections, over_detections)

    # Create and display confusion matrix
    create_confusion_matrix(ground_truth_list, predicted_list)

    print(f"Total Images: {total_images}")
    print(f"Correct Detections: {correct_detections}")
    print(f"Under Detections: {under_detections}")
    print(f"Over Detections: {over_detections}")
    print(f"Mismatch IDs saved in: {output_txt}")

    print(f"Precision (%): {precision:.2f}")
    print(f"Recall (%): {recall:.2f}")
    print(f"F1-score (%): {f1_score:.2f}")
    print(f"AP (%): {ap:.2f}")
