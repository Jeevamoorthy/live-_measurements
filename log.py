import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale (use a corrected path)
img_path = r'C:/Users/jeeva/OneDrive/Desktop/edge detection/WhatsApp Image 2025-10-01 at 08.49.41_6f508895.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # 1. Apply Gaussian Blur to reduce noise
    # The (3, 3) is the kernel size. It must be an odd number.
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # 2. Apply Laplacian operator
    # cv2.CV_64F is used to capture the full range of changes (positive and negative)
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)

    # Convert the output back to an 8-bit image for display
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Display the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(laplacian_abs, cmap='gray')
    plt.title('Laplacian of Gaussian (LoG)')
    plt.xticks([]), plt.yticks([])

    plt.show()