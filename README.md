#  A Comparative Study of Edge Detection Algorithms

This project presents a **comparative analysis and practical implementation** of three classic edge detection algorithms used in Computer Vision:

- **Canny Edge Detection**
- **Sobel Edge Detection**
- **Laplacian of Gaussian (LoG)**

The study focuses on understanding how different derivative-based methods identify intensity gradients and edges, and how they behave under noise and real-time conditions.

---

##  Sample Results


![1759313705125](https://githu![1759313704147](https://github.com/user-attachments/assets/496af06a-4eb5-4950-881c-b3ea63dc467d)
b.com/user-attachments/assets/c4d75a7f-59d7-423c-8369-53fbf4364b04)

---

##  Algorithms Implemented

### 1️ Canny Edge Detection
Canny is a **multi-stage, industry-standard algorithm** known for its robustness and low false detection rate.

**Implementation Pipeline:**
- **Gaussian Blur** (`cv2.GaussianBlur`)  
  Suppresses noise to prevent false edge detection.
- **Gradient Computation**  
  Uses Sobel derivatives internally to compute gradient magnitude and orientation.
- **Non-Maximum Suppression**  
  Thins edges to a single-pixel width.
- **Hysteresis Thresholding**  
  Uses high and low thresholds to retain strong edges and connect weak but relevant edges.

 *Result: Clean, continuous, and well-defined contours.*

---

### 2️ Sobel Edge Detection
A **first-order derivative operator** used for fast edge detection.

**Key Steps:**
- Applied `cv2.Sobel` separately along **X and Y axes**.
- Combined directional gradien
