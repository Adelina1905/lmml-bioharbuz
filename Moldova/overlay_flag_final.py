import cv2
import os
import numpy as np
import glob

FLAG_PATH = "input_flag.png"
EDITED_MANUAL = "BinaryMasks/edited"
EDITED_REST = "BinaryMasks/edited_all_except_manual"
EXTRA_MASK = "BinaryMasks/apple_22_mask_mrcnn.png"
OUTPUT_PATH = "BinaryMasks/flag_with_apples_masked_final.png"

# Load flag image
flag_img = cv2.imread(FLAG_PATH)
if flag_img is None:
    raise FileNotFoundError(f"Flag image not found: {FLAG_PATH}")

combined_mask = np.zeros((flag_img.shape[0], flag_img.shape[1]), dtype=np.uint8)

# Add all masks from edited manual folder
for mask_file in glob.glob(os.path.join(EDITED_MANUAL, '*_mask_mrcnn.png')):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask: {mask_file}")
        continue
    mask_resized = cv2.resize(mask, (flag_img.shape[1], flag_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

# Add all masks from edited rest folder
for mask_file in glob.glob(os.path.join(EDITED_REST, '*_mask_mrcnn.png')):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask: {mask_file}")
        continue
    mask_resized = cv2.resize(mask, (flag_img.shape[1], flag_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

# Add the extra mask
if os.path.exists(EXTRA_MASK):
    mask = cv2.imread(EXTRA_MASK, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        mask_resized = cv2.resize(mask, (flag_img.shape[1], flag_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
    else:
        print(f"Could not load mask: {EXTRA_MASK}")
else:
    print(f"Extra mask not found: {EXTRA_MASK}")

# Overlay: where mask is 255, set flag image to black
flag_masked = flag_img.copy()
flag_masked[combined_mask == 255] = 0

# Save result
cv2.imwrite(OUTPUT_PATH, flag_masked)
print(f"Final overlay of all masks on flag saved to {OUTPUT_PATH}")
