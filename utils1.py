import numpy as np
import math

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max  = box1[0][0], box1[0][1], box1[2][0], box1[2][1]
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area

    return iou

def find_max_overlap(bounding_boxes, reference_box):
    max_iou = 0
    max_overlap_box = None

    for box in bounding_boxes:
        iou = calculate_iou(box, reference_box)
        if iou > max_iou:
            max_iou = iou
            max_overlap_box = box

    return max_overlap_box, max_iou

def calculate_vertical_edge_angle(max_overlap_box):
    sorted_points = max_overlap_box[np.argsort(max_overlap_box[:, 1])]
    sorted_points = sorted_points[np.argsort(sorted_points[:, 0])]

    top_two = sorted_points[:2]
    bottom_two = sorted_points[2:]

    top_right = top_two[np.argmax(top_two[:, 0])]
    bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]

    dy = bottom_right[1] - top_right[1]
    dx = bottom_right[0] - top_right[0]

    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    angle_with_vertical = 90 - angle_degrees

    print(f"Angle of the right vertical edge with the right edge of the image: {angle_with_vertical:.2f} degrees")

    return angle_with_vertical
