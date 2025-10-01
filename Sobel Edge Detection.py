import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale (use a corrected path)
# Make sure to use forward slashes or a raw string
img_path = r'C:/Users/jeeva/OneDrive/Desktop/edge detection/WhatsApp Image 2025-10-01 at 08.49.41_6f508895.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Apply Sobel operator in X and Y directions
    # cv2.CV_64F is the data type to handle negative slopes (light to dark)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Gx (derivative in x direction)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Gy (derivative in y direction)

    # Convert back to an 8-bit unsigned integer
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combine the gradients to get the final edge image
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # Display the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(abs_sobel_x, cmap='gray')
    plt.title('Sobel X (Vertical Edges)')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(abs_sobel_y, cmap='gray')
    plt.title('Sobel Y (Horizontal Edges)')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Combined')
    plt.xticks([]), plt.yticks([])

    plt.show()