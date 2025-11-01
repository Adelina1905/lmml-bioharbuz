import cv2
import numpy as np
import os
# Load the image
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

img = cv2.imread(os.path.join(SCRIPT_PATH,"distorted_qr.png"))

# Convert to grayscale (optional, if you want to detect pure black easily)
black_mask = np.all(img == [0, 0, 0], axis=-1)

# Create a white background
result = np.ones_like(img) * 255

# Set black pixels in the result
result[black_mask] = [0, 0, 0]

# Save the result
cv2.imwrite("black_only.png", result)
