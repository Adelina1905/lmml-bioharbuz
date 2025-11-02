import cv2
import os
import numpy as np
import glob

FLAG_PATH = "input_flag.png"
MASKS_FOLDER = "binaryMarks/edited_all"
OUTPUT_PATH = "binaryMarks/flag_with_apples_masked_all.png"


# Load flag image
flag_img = cv2.imread(FLAG_PATH)
if flag_img is None:
    raise FileNotFoundError(f"Flag image not found: {FLAG_PATH}")

# Get all mask files in edited_all folder
mask_files = sorted(glob.glob(os.path.join(MASKS_FOLDER, '*.png')))
combined_mask = np.zeros((flag_img.shape[0], flag_img.shape[1]), dtype=np.uint8)
for mask_file in mask_files:
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask: {mask_file}")
        continue
    mask_resized = cv2.resize(mask, (flag_img.shape[1], flag_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_mask = cv2.bitwise_or(combined_mask, mask_resized)


# Overlay: where mask is 255, set flag image to black
flag_masked = flag_img.copy()
flag_masked[combined_mask == 255] = 0

# Save result
cv2.imwrite(OUTPUT_PATH, flag_masked)
print(f"Overlay of all masks in {MASKS_FOLDER} on flag saved to {OUTPUT_PATH}")
