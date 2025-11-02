import cv2
import os
import numpy as np


MASKS_DIR = "BinaryMasks"
OUTPUT_DIR = "BinaryMasks/edited_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all mask files
mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('_mask_mrcnn.png')]
mask_files.sort()

editing = False
mode = 'draw'  # 'draw' or 'erase'

def mouse_event(event, x, y, flags, param):
    global editing, mode, mask_img
    if event == cv2.EVENT_LBUTTONDOWN:
        editing = True
        mode = 'draw'
    elif event == cv2.EVENT_RBUTTONDOWN:
        editing = True
        mode = 'erase'
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        editing = False
    elif event == cv2.EVENT_MOUSEMOVE and editing:
        color = 255 if mode == 'draw' else 0
        cv2.circle(mask_img, (x, y), 10, color, -1)

for mask_name in mask_files:
    # Get corresponding apple image
    img_name = mask_name.replace('_mask_mrcnn.png', '.jpg').replace('_mask_mrcnn.png', '.png')
    img_path_jpg = os.path.join(MASKS_DIR.replace('BinaryMasks', 'Apples'), img_name.replace('.png', '.jpg'))
    img_path_png = os.path.join(MASKS_DIR.replace('BinaryMasks', 'Apples'), img_name.replace('.jpg', '.png'))
    if os.path.exists(os.path.join(OUTPUT_DIR, mask_name)):
        print(f"Already edited: {mask_name}, skipping.")
        continue
    if os.path.exists(img_path_jpg):
        apple_img = cv2.imread(img_path_jpg)
    elif os.path.exists(img_path_png):
        apple_img = cv2.imread(img_path_png)
    else:
        print(f"Could not find apple image for {mask_name}")
        continue
    h, w = apple_img.shape[:2]
    mask_img = np.zeros((h, w), dtype=np.uint8)  # always start with blank mask
    cv2.namedWindow("Edit Mask")
    cv2.setMouseCallback("Edit Mask", mouse_event)
    print(f"Editing {mask_name}: Left-click to draw, Right-click to erase, 's' to save, 'n' for next image, 'q' to quit.")
    while True:
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(mask_rgb, 0.5, apple_img, 0.5, 0)
        cv2.imshow("Edit Mask", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            out_path = os.path.join(OUTPUT_DIR, mask_name)
            cv2.imwrite(out_path, mask_img)
            print(f"Saved to {out_path}")
        elif key == ord('n'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Exited early.")
            exit(0)
    cv2.destroyAllWindows()
print("Done editing masks.")
