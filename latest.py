import cv2
import numpy as np

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================

# STEP 1: CALIBRATE THIS VALUE
# This is the most important setting. You must calculate the correct value for your setup.
PIXELS_PER_METRIC = 33.27 # Replace this example value with your calculated one

# STEP 2: SET YOUR CAMERA'S URL
# Start the IP Webcam app on your phone and enter the address it shows.
# Remember to add "/video" to the end of the address.
URL = "http://192.0.0.4:8080/video"  # <--- REPLACE THIS WITH YOUR PHONE'S IP

# STEP 3: DEFINE YOUR WORKING AREA (Optional but recommended)
# Format: (Top Y, Bottom Y, Left X, Right X)
ROI_COORDS = (100, 600, 50, 800) # Adjust these values to fit your camera's view

# STEP 4: SET DETECTION SENSITIVITY
# Increase to ignore shadows; decrease to detect fainter objects.
THRESHOLD_SENSITIVITY = 40 
MIN_CONTOUR_AREA = 1500    # Ignores any detected shapes smaller than this

cap = cv2.VideoCapture(URL) 

if not cap.isOpened():
    print("Error: Could not open video stream from URL.")
    print("Please check the URL and ensure your phone and computer are on the same Wi-Fi network.")
    exit()

# This will store our "empty" background image
background = None

print("Successfully connected to camera stream.")
print("INSTRUCTIONS:")
print("1. Position your camera.")
print("2. Make sure the ROI (yellow box) is empty.")
print("3. Press 'b' to capture the background.")
print("4. Place an object in the ROI to measure it.")
print("5. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from stream. Check Wi-Fi connection.")
        break

    # Define and draw the Region of Interest (ROI) on the frame
    y1, y2, x1, x2 = ROI_COORDS
    frame_h, frame_w, _ = frame.shape
    y2 = min(y2, frame_h)
    x2 = min(x2, frame_w)
    
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Wait for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        # Capture the background when 'b' is pressed
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        print("Background set!")
        continue

    # Once background is set, start detecting changes
    if background is not None:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (21, 21), 0)

        # Calculate the difference between the background and the current frame
        diff_frame = cv2.absdiff(background, gray_roi_blurred)

        # Threshold the difference image to get a clean mask of the new object
        thresh = cv2.threshold(diff_frame, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        
        # --- Clean up the mask to remove noise and improve accuracy ---
        kernel = np.ones((5, 5), np.uint8)
        cleaned_thresh = cv2.erode(thresh, kernel, iterations=1)
        cleaned_thresh = cv2.dilate(cleaned_thresh, kernel, iterations=5)
        
        # Find contours of the new object(s) using the cleaned mask
        contours, _ = cv2.findContours(cleaned_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue

            # Get the rotated bounding box for the object
            rect = cv2.minAreaRect(c)
            box_roi = cv2.boxPoints(rect)
            box_roi = np.intp(box_roi)

            # --- Measurement Logic ---
            width_px, height_px = rect[1]
            length_px = max(width_px, height_px)
            breadth_px = min(width_px, height_px)

            # --- CALIBRATION STEP ---
            # To calibrate, uncomment the line below, place a ruler in the ROI,
            # and use the printed pixel length to calculate your PIXELS_PER_METRIC value.
            print(f"PIXEL CHECK -> Length: {length_px:.2f} px")
            # ------------------------

            length_cm = length_px / PIXELS_PER_METRIC
            breadth_cm = breadth_px / PIXELS_PER_METRIC

            # Convert ROI box coordinates to full frame coordinates for drawing
            box_full_frame = box_roi + (x1, y1)

            # Draw the box and measurements on the main frame
            cv2.drawContours(frame, [box_full_frame], 0, (0, 255, 0), 2)
            cv2.putText(frame, f"L: {length_cm:.1f}cm", (box_full_frame[0][0], box_full_frame[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"B: {breadth_cm:.1f}cm", (box_full_frame[0][0], box_full_frame[0][1] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Display the mask for debugging
        cv2.imshow("Change Mask", cleaned_thresh)

    # Display the final output
    cv2.imshow("Live Measurement (Mobile Camera)", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()