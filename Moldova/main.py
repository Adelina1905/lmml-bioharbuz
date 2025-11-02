#from ultralytics import YOLO
import cv2
import numpy as np
import os

def overlay_masks_on_flag(flag_path, masks_folder, output_path):
    import glob
    # Load flag image
    flag_img = cv2.imread(flag_path)
    if flag_img is None:
        raise FileNotFoundError(f"Flag image not found: {flag_path}")

    # Get all mask files
    mask_files = sorted(glob.glob(os.path.join(masks_folder, '*_mask_mrcnn.png')))
    if not mask_files:
        print("No mask files found.")
        return

        # Masks to use from edited folder
        edited_names = {"apple_1_mask_mrcnn.png", "apple_4_mask_mrcnn.png", "apple_7_mask_mrcnn.png", "apple_11_mask_mrcnn.png", "apple_13_mask_mrcnn.png", "apple_25_mask_mrcnn.png"}

        combined_mask = np.zeros((flag_img.shape[0], flag_img.shape[1]), dtype=np.uint8)
        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file)
            if mask_name in edited_names:
                mask_path = os.path.join(edited_folder, mask_name)
                if not os.path.exists(mask_path):
                    print(f"Edited mask not found for {mask_name}, using automatic.")
                    mask_path = mask_file
            else:
                mask_path = mask_file
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_resized = cv2.resize(mask, (flag_img.shape[1], flag_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Overlay: where mask is 255, set flag image to black
        flag_masked = flag_img.copy()
        flag_masked[combined_mask == 255] = 0

        # Save result
        cv2.imwrite(output_path, flag_masked)
        print(f"Overlay of selected masks on flag saved to {output_path}")

    # --- Post-processing: keep largest, roundest regions ---
    mask_proc = combined_mask_np.copy()
    contours, _ = cv2.findContours(mask_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask_proc)
    # Stricter settings for specific images
    hard_mode_files = {"apple_1.jpg", "apple_4.jpg", "apple_7.jpg", "apple_11.jpg", "apple_13.png", "apple_25.png"}
    area_thresh = 500
    round_thresh = 0.5
    if fname in hard_mode_files:
        area_thresh = 1200  # increase area threshold
        round_thresh = 0.7  # increase roundness threshold
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        roundness = 4 * np.pi * area / (perimeter * perimeter)
        if roundness > round_thresh:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    # Save binary mask
    cv2.imwrite(output_mask, filtered_mask)
    print(f"Mask R-CNN mask saved to {output_mask}")

    # Overlay mask on image for visualization
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    mask_resized = cv2.resize(filtered_mask, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    alpha = 0.9
    overlay = cv2.addWeighted(mask_rgb, alpha, img_cv, 1 - alpha, 0)
    cv2.imwrite(output_overlay, overlay)
    print(f"Mask R-CNN overlay saved to {output_overlay}")
    overlay_selected_masks_on_flag(
        flag_path="input_flag.png",
        masks_folder="BinaryMasks",
        edited_folder="BinaryMasks/edited",
        output_path="BinaryMasks/flag_with_apples_masked.png"
    )

# Install ultralytics before running:
# pip install ultralytics




import torch
import torchvision
from torchvision import transforms
from PIL import Image

apples_folder = "Apples"
output_folder = "BinaryMasks"
os.makedirs(output_folder, exist_ok=True)

# Load pre-trained Mask R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

for fname in os.listdir(apples_folder):
    if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
        continue
    input_image_path = os.path.join(apples_folder, fname)
    output_mask = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask_mrcnn.png")
    output_overlay = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_overlay_mrcnn.png")

    # Load and preprocess image
    img_pil = Image.open(input_image_path).convert("RGB")
    img_tensor = transform(img_pil).to(device)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Select masks for apples (COCO class 53: apple)
    masks = prediction['masks']
    labels = prediction['labels']
    apple_indices = [i for i, label in enumerate(labels) if label.item() == 53]
    if not apple_indices:
        print(f"No apples detected in {fname}.")
        continue

    # Combine all apple masks
    combined_mask = torch.zeros_like(masks[0][0], dtype=torch.uint8)
    for i in apple_indices:
        mask = (masks[i][0] > 0.5).to(torch.uint8)
        combined_mask = torch.logical_or(combined_mask, mask)
    combined_mask_np = combined_mask.cpu().numpy() * 255

    # Save binary mask
    cv2.imwrite(output_mask, combined_mask_np)
    print(f"Mask R-CNN mask saved to {output_mask}")

    # Overlay mask on image for visualization
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    mask_resized = cv2.resize(combined_mask_np, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    overlay = cv2.addWeighted(mask_rgb, alpha, img_cv, 1 - alpha, 0)
    cv2.imwrite(output_overlay, overlay)
    print(f"Mask R-CNN overlay saved to {output_overlay}")
