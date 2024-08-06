import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np
import json
import os


def crop_image(image, box):
    return image.crop(box)


def grounding_dino(image_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Replace this with your local image path
    # image_path = "/path/to/your/image.jpg"
    # image_path = "/home/ammara/Documents/helper_code/extracted_frames/movie_3/frame_0.jpg"
    image = Image.open(image_path)

    # Check for cats and remote controls
    text = "person. golf stick. door."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    print(results)

    # Find all 'golf stick' boxes
    golf_stick_boxes = [(box.cpu().numpy(), score.item()) 
                        for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']) 
                        if label == 'golf stick']

    # Create a directory to save results
    os.makedirs("results", exist_ok=True)

    # Crop images for each golf stick box
    for i, (box, score) in enumerate(golf_stick_boxes):
        # Convert box coordinates to integers
        box = [int(coord) for coord in box]
        cropped_image = crop_image(image, box)
        
        # Save the cropped image
        image_filename = f"golf_stick_{i+1}.jpg"
        cropped_image.save(os.path.join("results", image_filename))
        
        # Save the box information
        box_info = {
            "box": box,
            "score": score,
            "label": "golf stick"
        }
        json_filename = f"golf_stick_{i+1}_info.json"
        with open(os.path.join("results", json_filename), 'w') as f:
            json.dump(box_info, f, indent=4)

    print(f"Cropped {len(golf_stick_boxes)} golf stick images and saved their information.")   

    # Find all 'door' boxes
    door_boxes = [(box.cpu().numpy(), score.item()) 
                    for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']) 
                    if label == 'door']

    # Crop images for each golf stick box
    for i, (box, score) in enumerate(door_boxes):
        # Convert box coordinates to integers
        box = [int(coord) for coord in box]
        cropped_image = crop_image(image, box)
        
        # Save the cropped image
        image_filename = f"door_{i+1}.jpg"
        cropped_image.save(os.path.join("results", image_filename))
        
        # Save the box information
        box_info = {
            "box": box,
            "score": score,
            "label": "door"
        }
        json_filename = f"door_{i+1}_info.json"
        with open(os.path.join("results", json_filename), 'w') as f:
            json.dump(box_info, f, indent=4)

    print(f"Cropped {len(door_boxes)} door images and saved their information.")    


if __name__ == "__main__":
    image_path = "/home/ammara/Documents/helper_code/extracted_frames/movie_3/frame_0.jpg"
    grounding_dino(image_path)
