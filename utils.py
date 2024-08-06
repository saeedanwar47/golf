import numpy as np 
import cv2 
import math
import matplotlib.pyplot as plt


# calculate the overlap between box points and golf bbox, find the box with maximum overlap 
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: list of four integers [xmin, ymin, xmax, ymax]

    Returns:
    float: IoU value
    """
    # Unpack coordinates
    x1_min, y1_min, x1_max, y1_max  = box1[0][0], box1[0][1], box1[2][0], box1[2][1]
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection coordinates
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    # Calculate area of intersection
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Calculate area of each bounding box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    
    return iou


def find_max_overlap(bounding_boxes, reference_box):
    """
    Find the bounding box with the maximum overlap with the reference box.

    Parameters:
    bounding_boxes: list of lists, where each inner list represents a bounding box [xmin, ymin, xmax, ymax]
    reference_box: list of four integers [xmin, ymin, xmax, ymax]

    Returns:
    tuple: (bounding box with max overlap, IoU value)
    """
    max_iou = 0
    max_overlap_box = None
    
    for box in bounding_boxes:
        iou = calculate_iou(box, reference_box)
        if iou > max_iou:
            max_iou = iou
            max_overlap_box = box
    
    return max_overlap_box, max_iou


def calculate_vertical_edge_angle(max_overlap_box):
    # calculate the angle of the left vertical edge of bbox with vertical axis of image
    sorted_points = max_overlap_box[np.argsort(max_overlap_box[:, 1])]
    sorted_points = sorted_points[np.argsort(sorted_points[:, 0])]

    # Identify the top-right and bottom-right vertices
    # top-right: the point with maximum x-coordinate from the top two points
    # bottom-right: the point with maximum x-coordinate from the bottom two points
    top_two = sorted_points[:2]
    bottom_two = sorted_points[2:]

    top_right = top_two[np.argmax(top_two[:, 0])]
    bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]

    # Calculate the differences in coordinates
    dy = bottom_right[1] - top_right[1]
    dx = bottom_right[0] - top_right[0]

    # Calculate the angle in radians
    angle_radians = math.atan2(dy, dx)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    # Since we need the angle with the vertical edge (y-axis), we take 90 degrees minus the calculated angle
    angle_with_vertical = 90 - angle_degrees

    print(f"Angle of the right vertical edge with the right edge of the image: {angle_with_vertical:.2f} degrees")

    return angle_with_vertical


