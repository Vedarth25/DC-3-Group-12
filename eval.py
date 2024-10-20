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
from Preprocessing_class import *
from deep_sort_realtime.deepsort_tracker import DeepSort

"""hyper parameters"""
use_cuda = False


def detect_fish_in_image(imgfile, m, deepsort, unique_fish_ids, pre=False):
    """
    Function to detect fish in a given image using YOLO model.
    Uses Deep SORT to track and assign unique IDs to fish.
    """
    img = cv2.imread(imgfile)
    if pre:
        p = ImagePreprocessor('data', target_size=(m.width, m.height)) 
        sized = p.preprocess_image(img)
    else:
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # YOLO detection
    boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)

    # Convert YOLO bounding boxes (x_center, y_center, width, height) into (left, top, width, height) format for Deep SORT
    detections = []  

    for box in boxes[0]:
        x_center, y_center, width, height = box[0], box[1], box[2], box[3]
        left = int((x_center - width / 2) * m.width)  # left = x_center - (width / 2)
        top = int((y_center - height / 2) * m.height)  # top = y_center - (height / 2)
        w = int(width * m.width)  # Convert width to pixel space
        h = int(height * m.height)  # Convert height to pixel space
        
        # Combine the bounding box and confidence score in one tuple
        detections.append(([left, top, w, h], box[4], 0))  # Using '0' as a placeholder class for fish

    
    print(f"detections: {detections}")  # This is just for debugging: print the detections

    if len(detections) > 0:
        # Update Deep SORT tracker with detections and the image frame (passing the frame for internal embedding)
        outputs = deepsort.update_tracks(detections, frame=img)

        # Add unique track IDs to the set
        for output in outputs:
            track_id = output.track_id  # Each output contains a track_id attribute
            unique_fish_ids.add(track_id)  

        return len(outputs), boxes, outputs  
    else:
        print("Error: Detection format is incorrect or empty!") 
        return 0, boxes, []









    # Assuming that each box is a fish detection
    #num_fish_detected = len(boxes[0])  # Each box is a fish detection

    #return num_fish_detected, boxes



def evaluate_model(csv_file, image_folder, output_folder, m, deepsort, unique_fish_ids, pre=False):
    """
    Function to evaluate the YOLO model against images from the CSV file.
    Records mismatches, tracks over/under detection for the confusion matrix.
    """
    
    data = pd.read_csv(csv_file)
    total_images = len(data)

    mismatch_ids = []
    final_tp = 0
    final_fn = 0
    final_fp = 0

    
    for index, row in tqdm(data.iterrows(), total=total_images, desc="Evaluating images", unit="img"):
        img_id = row['ID']
        ground_truth_count = row['counts']
        #TODO make a logic that extracts the habbitat and initialises a tracker for that specific habbitat

        
        img_path = os.path.join(image_folder, img_id + '.jpg')

        
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found!")
            continue

        # Detect fish in the image and update Deep SORT
        detected_count, boxes, tracked_objects = detect_fish_in_image(img_path, m, deepsort, unique_fish_ids, pre)

        # Load the mask to compare with the detected boxes
        mask_path = os.path.join('data/masks', img_id + '.png')
        mask = cv2.imread(mask_path)
        tp, fp, fn = tp_fp_fn(boxes[0], mask)

        # Log the ground truth count and detected count
        print(f"Image {img_id}: Ground truth count = {ground_truth_count}, Detected count = {detected_count}")
        
        # Check for correctness, log mismatches
        if tp + len(fn) != ground_truth_count or tp + len(fp) != detected_count:
            print(f"Mismatch for image {img_id}: Ground truth count = {ground_truth_count}, "
                  f"Detected count = {detected_count}, True Positives = {tp}, False Positives = {len(fp)}, False Negatives = {len(fn)}")
            mismatch_ids.append(f"{img_id} FP: {len(fp)}, FN: {len(fn)}")
            continue  # Skip to the next image without raising an exception
        
        # Accumulate TP, FP, FN counts
        final_tp += tp
        final_fp += len(fp)
        final_fn += len(fn)

    # Save mismatches to a txt file
    with open(f"{output_folder}mismatch_ids.txt", 'w') as f:
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
    # Sort in order by x
    boxes.sort(key=lambda box: box[0])
    points = sorted(points, key=lambda point: cv2.minEnclosingCircle(point)[0][0])
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
            if x1 <= x <= x2 and y1 <= y <= y2 and center not in covered_points:
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




import os

def evaluate_from_test_folder(test_folder, m, deepsort, unique_fish_ids, pre=False, output_folder='demo_images/'):
    """
    Evaluate the YOLO model on a test folder of images with associated text files.
    Visualize the first 25% of the dataset and save the images in the 'demo_images' folder.
    """
    # Ensure the output folder 'demo_images/' exists or make it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    total_images = len(image_files)

    final_tp = 0
    final_fp = 0
    final_fn = 0

    mismatch_ids = []
    all_ground_truths = []
    all_predictions = []

    for img_idx, img_file in tqdm(enumerate(image_files), total=total_images, desc="Evaluating images from test folder", unit="img"):
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
            x1 = (center_x - width / 2)
            y1 = (center_y - height / 2)
            x2 = (center_x + width / 2)
            y2 = (center_y + height / 2)
            ground_truth_boxes.append([x1, y1, x2, y2])

        # Detect fish in the image and track them using Deep SORT
        detected_count, boxes, tracked_objects = detect_fish_in_image(img_path, m, deepsort, unique_fish_ids, pre)

        # Log the ground truth count and detected count
        print(f"Image {img_id}: Ground truth count = {ground_truth_count}, Detected count = {detected_count}")
        
        # Match bounding boxes (ground truth vs detected)
        tp, fp, fn = match_bboxes(ground_truth_boxes, boxes[0], iou_threshold=0.5)

        # Log mismatch if there's any
        if tp + fn != ground_truth_count or tp + fp != detected_count:
            print(f"Mismatch for image {img_id}: Ground truth count = {ground_truth_count}, "
                  f"Detected count = {detected_count}, True Positives = {tp}, False Positives = {fp}, False Negatives = {fn}")
            mismatch_ids.append(f"{img_id} FP: {fp}, FN: {fn}")
            continue  # Skip to the next image without raising an exception
        
        final_tp += tp
        final_fp += fp
        final_fn += fn

        # Visualize tracks for the first 50% of the dataset
        if img_idx < total_images * 0.5:  
            save_path = f"{output_folder}tracked_frame_{img_id}.jpg"
            modified_image = visualize_tracks(cv2.imread(img_path), tracked_objects, save_path)  # Visualize the results

    # Save mismatches to a txt file
    with open(f"{output_folder}mismatch_test_ids.txt", 'w') as f:
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
    
    #sort boxes by x values
    true_bboxes.sort(key=lambda box: box[0])
    pred_bboxes.sort(key=lambda box: box[0])
    boxes = true_bboxes + pred_bboxes
    print("amount of predicted boxes", len(pred_bboxes))
    for pred_box in pred_bboxes:
        matched = False
        max = 0
        print("predicted box",pred_box)
        
        for i, true_box in enumerate(true_bboxes):
            iou = bbox_iou(true_box, pred_box)
            print(iou, true_box)
            if iou >= iou_threshold and not matched_true_boxes[i] and iou > max:
                max = iou
                max_index = i
        # It's a match!
        if max > 0:
            print(max)
            TP += 1
            matched_true_boxes[max_index] = True
            matched = True
        else:
            # If no true box was matched, it's a false positive
            FP += 1
    
    # Any unmatched true box is a false negative
    FN = matched_true_boxes.count(False)
    
    return TP, FP, FN



def visualize_tracks(image, tracked_objects, save_path):
    """
    Visualize bounding boxes and track IDs on the image and return the modified image.
    """
    for track in tracked_objects:
        # Convert track coordinates to integers
        left, top, right, bottom = map(int, track.to_ltrb())  # Ensure coordinates are integers
        track_id = track.track_id

        # Draw bounding box and track ID on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f'ID: {track_id}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the modified image
    cv2.imwrite(save_path, image)

    return image











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

def create_confusion_matrix_from_counts(TP, FP, FN, output_folder):
    """
    Function to create and display/save a confusion matrix based on counts of TP, FP, and FN.
    Excludes True Negatives (TN) since they are not tracked.
    """
 
    # We assume TN is unknown, so we only show counts for TP, FP, and FN.
    confusion_matrix = [[TP, FP],   # Row for "Predicted Positive"
                        [FN, 0]]    # Row for "Predicted Negative" (TN is unknown, set to 0)

    # Define labels for the axes
    labels = ['Predicted Positive', 'Predicted Negative']
    categories = ['Actual Positive', 'Actual Negative']

    plt.figure(figsize=(8, 6))
    
    # Create a heatmap with annotations of the confusion matrix values
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=categories, yticklabels=labels)
    
    plt.title('Confusion Matrix (Excludes TN)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    # Save and display the confusion matrix plot
    plt.savefig(f'{output_folder}confusion_matrix.png')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-p', '--preprocess', action='store_true',
                        help='enable pre-processing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='make it test agains test folder')
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
    output_folder = 'res_no_preproc/'
    args = get_args()
    if args.preprocess:
        output_folder = "res_preproc/"

    m = Darknet(cfgfile)
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()
    
    # Initialize Deep SORT 
    #TODO redo tracker for each habitat instead
    deepsort = DeepSort(max_age=10, n_init=2, nms_max_overlap=1.0, embedder_gpu=use_cuda)

    # Initialize the set to track unique fish IDs 
    unique_fish_ids = set()


    if args.test:
        TP, FP, FN = evaluate_from_test_folder('data/test', m, deepsort, unique_fish_ids, args.preprocess, output_folder)
    else:
        TP, FP, FN = evaluate_model(csv_file, image_folder, output_folder, m, deepsort, unique_fish_ids, args.preprocess)
    
    precision, recall, f1_score = calculate_metrics(TP, FP, FN)


    # Create the infographic
    #create_infographic(total_images, correct_detections, under_detections, over_detections)

    # Create and display confusion matrix
    create_confusion_matrix_from_counts(TP, FP, FN, output_folder)



    print(f"Mismatch IDs saved in: {output_folder}")

    print(f"Precision (%): {precision:.2f}")
    print(f"Recall (%): {recall:.2f}")
    print(f"F1-score (%): {f1_score:.2f}")

    print(f"Total number of unique fishes: {len(unique_fish_ids)}")
    #print(f"AP (%): {ap:.2f}")
