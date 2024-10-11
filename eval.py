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

    return num_fish_detected, boxes

def evaluate_model(csv_file, image_folder, output_txt, m, pre=False):
    """
    Function to evaluate the YOLO model against images from the CSV file.
    Records the mismatches and saves the result in a text file.
    Also tracks over-detection, under-detection for confusion matrix.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    total_images = len(data)

    mismatch_ids = []
    final_tp = 0
    final_fn = 0
    final_fp = 0

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
        detected_count, boxes = detect_fish_in_image(img_path, m, pre)

        # Load the mask to compare with the detected boxes
        mask_path = os.path.join('data/masks', img_id + '.png')
        mask = cv2.imread(mask_path)
        tp, fp, fn = tp_fp_fn(boxes[0], mask)
        #check for correctness
        if tp + len(fn) != ground_truth_count or tp + len(fp) != detected_count:
            print(tp, len(fn), ground_truth_count, detected_count, tp, len(fp))
            raise Exception(f"Ground truth count mismatch for image {img_id}. Your point matcher is shit")

        if len(fp) > 0 or len(fn) > 0:
            mismatch_ids.append(img_id +  " FP:" + str(len(fp)) + " FN" + str(len(fn)))
        
        final_tp += tp
        final_fp += len(fp)
        final_fn += len(fn)

    # Save mismatches to a txt file
    with open(output_txt, 'w') as f:
        for img_id in mismatch_ids:
            f.write(f"{img_id}\n")

    return final_tp, final_fp, final_fn

def tp_fp_fn(boxes, mask):
    """
    Function to calculate true positives, false positives, and false negatives from boxes and points
    """
    # Convert mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    points, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tp = 0
    fp = []
    fn = []
    covered_points = set()

    height, width = mask.shape[:2]
    #check for true positives and false positives
    print(boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        box_found = False


        for point in points:
            (x, y), _ = cv2.minEnclosingCircle(point)
            center = (int(x), int(y))
            if x1 <= x <= x2 and y1 <= y <= y2:
                tp += 1
                box_found = True
                covered_points.add(center)
                break
        if not box_found:
            fp.append(box)

    #check for false negatives
    for point in points:
        (x, y), _ = cv2.minEnclosingCircle(point)
        center = (int(x), int(y))
        if tuple(center) not in covered_points:
            fn.append(center)
    
    return tp, fp, fn

def evaluate_from_test_folder(test_folder, m, pre=False):
    """
    Evaluate the YOLO model on a test folder of images with associated text files.
    """
    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    total_images = len(image_files)

    final_tp = 0
    final_fp = 0
    final_fn = 0

    mismatch_ids = []
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
            lines = f.readlines()
        ground_truth_count = len(lines)  # Each line represents one object
        ground_truth_boxes = []
        for line in lines:
            class_id, center_x, center_y, width, height = map(float, line.split())
            x1 = int((center_x - width / 2) * m.width)
            y1 = int((center_y - height / 2) * m.height)
            x2 = int((center_x + width / 2) * m.width)
            y2 = int((center_y + height / 2) * m.height)
            ground_truth_boxes.append([x1, y1, x2, y2])

        detected_count, boxes = detect_fish_in_image(img_path, m, pre)

        tp, fp, fn = match_bboxes(ground_truth_boxes, boxes[0], iou_threshold=0.5)

        if tp + len(fn) != ground_truth_count or tp + len(fp) != detected_count:
            raise Exception(f"Ground truth count mismatch for image {img_id}. Your box matching is shit")
        
        if len(fp) > 0 or len(fn) > 0:
            mismatch_ids.append(img_id +  " FP:" + str(len(fp)) + " FN" + str(len(fn)))

        final_tp += tp
        final_fp += len(fp)
        final_fn += len(fn)

    # Save mismatches to a txt file
    with open("missmatched_IDs_test", 'w') as f:
        for img_id in mismatch_ids:
            f.write(f"{img_id}\n")

    return final_tp, final_fp, final_fn

def match_bboxes(true_bboxes, pred_bboxes, iou_threshold=0.5):
    """
    Matches ground truth boxes with predicted boxes using IoU.
    Returns TP, FP, FN counts.
    """
    TP = 0
    FP = 0
    FN = 0

    # Create an array to keep track of matched true bboxes
    matched_true_boxes = [False] * len(true_bboxes)
    
    for pred_box in pred_bboxes:
        matched = False
        
        for i, true_box in enumerate(true_bboxes):
            iou = bbox_iou(true_box, pred_box)
            if iou >= iou_threshold and not matched_true_boxes[i]:
                # It's a match!
                TP += 1
                matched_true_boxes[i] = True
                matched = True
                break
        
        if not matched:
            # If no true box was matched, it's a false positive
            FP += 1
    
    # Any unmatched true box is a false negative
    FN = matched_true_boxes.count(False)
    
    return TP, FP, FN

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

def create_confusion_matrix_from_counts(TP, FP, FN):
    """
    Function to create and display/save a confusion matrix based on counts of TP, FP, and FN.
    Excludes True Negatives (TN) since they are not tracked.
    """
    # Define a confusion matrix with 2 rows (Actual Positive and Negative) and 2 columns (Predicted Positive and Negative)
    # We assume TN is unknown, so we only show counts for TP, FP, and FN.
    confusion_matrix = [[TP, FP],   # Row for "Actual Positive"
                        [FN, 0]]    # Row for "Actual Negative" (TN is unknown, set to 0)

    # Define labels for the axes
    labels = ['Predicted Positive', 'Predicted Negative']
    categories = ['Actual Positive', 'Actual Negative']

    plt.figure(figsize=(8, 6))
    
    # Create a heatmap with annotations of the confusion matrix values
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=categories)
    
    plt.title('Confusion Matrix (Excludes TN)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    # Save and display the confusion matrix plot
    plt.savefig('confusion_matrix_from_counts_without_TN.png')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-p', '--preprocess', action='store_true',
                        help='enable pre-processing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='make it test agains test folder')
    parser.add_argument('-p')
    args = parser.parse_args()
    return args

#Does not calculate average precision
def calculate_metrics(TP, FP, FN):
    """
    Calculate precision, recall, F1-score, and average precision from TP, FP, FN.
    TP: True Positives
    FP: False Positives
    FN: False Negatives
    """
    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # Calculate F1-score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average Precision (AP) is typically calculated over the entire precision-recall curve,
    # but if you want a simple version, you can approximate it as:
    #average_precision = precision  # As a simple approximation, you can use precision.

    # Return metrics in percentages
    return precision * 100, recall * 100, f1_score * 100, #average_precision * 100


if __name__ == '__main__':
    # Configuration files
    cfgfile = 'yolo-fish-2.cfg'
    weightfile = 'merge_yolo-fish-2.weights'
    csv_file = 'data/Localization.csv'
    image_folder = 'data/images'
    output_txt = 'mismatch_ids.txt'
    args = get_args()



    m = Darknet(cfgfile)
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    if args.test:
        TP, FP, FN = evaluate_from_test_folder('test', m, args.preprocess)
    else:
        TP, FP, FN = evaluate_model(csv_file, image_folder, output_txt, m, args.preprocess)
    
    precision, recall, f1_score = calculate_metrics(TP, FP, FN)

    # Create the infographic
    #create_infographic(total_images, correct_detections, under_detections, over_detections)

    # Create and display confusion matrix
    create_confusion_matrix_from_counts(TP, FP, FN)

    print(f"Mismatch IDs saved in: {output_txt}")

    print(f"Precision (%): {precision:.2f}")
    print(f"Recall (%): {recall:.2f}")
    print(f"F1-score (%): {f1_score:.2f}")
    #print(f"AP (%): {ap:.2f}")
