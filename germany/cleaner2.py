import cv2
import numpy as np

# Alternative method: More aggressive filling
def aggressive_fill_qr(input_path, output_path):
    """
    More aggressive approach - dilates black regions to fill gaps
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Dilate black regions (expands black areas)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(cv2.bitwise_not(binary), kernel, iterations=1)
    
    # Apply closing to connect nearby regions
    kernel2 = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel2, iterations=1)
    
    result = cv2.bitwise_not(closed)
    
    cv2.imwrite(output_path, result)
    print(f"Aggressively filled QR code saved to {output_path}")
    
    return result

# Usage
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    input_image = os.path.join(SCRIPT_PATH,"black_only.png")
    
    # Try both methods
    result2 = aggressive_fill_qr(input_image, "cleaned_qr_method2.png")
    cv2.imwrite("black_only_cleaned.png", result2)

    # # Display comparison
    # try:
    #     from matplotlib import pyplot as plt
        
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
    #     original = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    #     axes[0].imshow(original, cmap='gray')
    #     axes[0].set_title('Original')
    #     axes[0].axis('off')
        
    #     axes[1].imshow(result1, cmap='gray')
    #     axes[1].set_title('Method 1: Neighborhood Filling')
    #     axes[1].axis('off')
        
    #     axes[2].imshow(result2, cmap='gray')
    #     axes[2].set_title('Method 2: Aggressive Filling')
    #     axes[2].axis('off')
        
    #     plt.tight_layout()
    #     plt.show()
    # except ImportError:
    #     print("Matplotlib not available")