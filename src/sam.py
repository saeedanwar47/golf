from PIL import Image 
from transformers import SamModel, SamProcessor
import torch 
import matplotlib.pyplot as plt 

def get_image_center(image):
    height, width = image.size[:2]
    center_y = height // 2
    center_x = width // 2
    return center_x, center_y

def sam(image_path, save_fig):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    # Replace this with the path to your local image
    # image_path = "/home/ammara/Documents/golf/notebooks/door_1.jpg"

    # Open the image from the local file system
    raw_image = Image.open(image_path).convert("RGB")

    # Calculate the center point
    center_x, center_y = get_image_center(raw_image)
    print(f"Image center: ({center_x}, {center_y})")

    # hardcode centre points
    # center_x = 100 
    # center_y = 100

    # Create the input_points with the center point
    input_points = [[[float(center_x), float(center_y)]]]
    print("Input points:", input_points)

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores 

    mask = masks[0][0].cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]  # Select the first channel if it's a 3D array

    plt.imshow(raw_image)
    plt.imshow(mask, alpha=0.6)
    # plt.axis('off')
    plt.savefig(f'results/{save_fig}.jpg')
    # plt.show()

    


if __name__ == "__main__":
    # this will be the image path of the cropped image 
    image_path = "results/door_1.jpg"
    save_fig = "door_mask"
    sam(image_path, save_fig)