import cv2
import numpy as np
import os

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
PIXELS_PER_METRIC = 19.33  # Pixels per Centimeter (ADJUST THIS)
INPUT_VIDEO_PATH = "C:/Users/jeeva/OneDrive/Desktop/edge detection/sample.mp4"
OUTPUT_VIDEO_PATH = "C:/Users/jeeva/OneDrive/Desktop/edge detection/output_video_kmeans.mp4"

# The number of distinct color clusters to find. This is a key parameter.
# More clusters can separate more colors but might split a single object.
NUMBER_OF_CLUSTERS = 8 

# Ignores any detected objects smaller than this value (in pixels^2).
MIN_CONTOUR_AREA = 1500

# =============================================================================
# --- VIDEO PROCESSING SCRIPT (K-MEANS METHOD) ---
# =============================================================================

def process_video_with_kmeans(input_path, output_path, ppm, k, min_area):
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at '{input_path}'")
        return

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Starting K-Means Color Segmentation video processing...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- K-MEANS COLOR SEGMENTATION ---

        # 1. Prepare the image for K-Means
        # Convert the image from BGR to RGB, as K-Means works with standard RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Reshape the image to be a list of pixels (N_pixels, 3)
        pixel_values = rgb_frame.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # 2. Apply K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape the labels back to the original image dimensions
        labels = labels.reshape((rgb_frame.shape[0], rgb_frame.shape[1]))

        # --- Loop through each color cluster to find and measure objects ---
        for i in range(k):
            # 3. Create a mask for the current cluster
            mask = np.zeros(labels.shape, dtype=np.uint8)
            mask[labels == i] = 255
            
            # You can uncomment the line below to see each mask individually
            # cv2.imshow(f"Mask {i}", mask)

            # 4. Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            # 5. Find contours on the cleaned mask
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < min_area:
                    continue

                # --- Measurement logic (same as before) ---
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                width_px, height_px = rect[1]
                length_px = max(width_px, height_px)
                breadth_px = min(width_px, height_px)
                
                length_cm = length_px / ppm
                breadth_cm = breadth_px / ppm

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                tl_corner = min(box, key=lambda p: p[0] + p[1])
                cv2.putText(frame, f"L:{length_cm:.1f}", (tl_corner[0], tl_corner[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, f"B:{breadth_cm:.1f}", (tl_corner[0], tl_corner[1] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # --- Save and display ---
        out.write(frame)
        cv2.imshow("Live Processing (K-Means)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nProcessing complete.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully saved to '{output_path}'")

# --- Main execution block ---
if __name__ == "__main__":
    process_video_with_kmeans(
        input_path=INPUT_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        ppm=PIXELS_PER_METRIC,
        k=NUMBER_OF_CLUSTERS,
        min_area=MIN_CONTOUR_AREA
    )