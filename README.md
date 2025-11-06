# 🚀 Project Overview

## This project aims to use computer vision techniques (OpenCV + HSV color segmentation) to:

1. Detect fish automatically in an image.

2. Extract contours and bounding boxes for shape analysis.

3. Calibrate pixel-to-centimeter ratios using a known reference object.

4. Compute estimated fish dimensions and surface area.

5. (working on it) Predict fish weight using a trained regression model.

The workflow currently focuses on catfish, with images sourced directly from a real fish farm for applied aquaculture use.

# 🧠 Key Features (Current Stage)

✅ HSV Color Segmentation
Segment fish from the background using the Hue–Saturation–Value color model.
Interactive HSV Tuner included to fine-tune thresholds per lighting condition.

✅ Contour Detection & Visualization
Extract the largest contour (fish body) and visualize it with bounding boxes.

✅ Bounding Box Measurement
Compute the width and height of the detected contour in pixels.

✅ Modular Code Design
Separate code chunks for HSV tuning, contour detection, and visualization.

# 🧩 Technical Workflow
### 1. Image Acquisition

Images can be captured directly from farm sites (e.g., catfish harvests) using a standard mobile camera under natural lighting.

### 2. HSV Color Segmentation

Convert the image from RGB to HSV space to isolate the fish body color range

```hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_bound, upper_bound)
```

A dedicated HSV Tuner script enables real-time adjustment of thresholds:
#### Example lower/upper bounds
```lower = np.array([0, 30, 60])
upper = np.array([20, 255, 255])
```
<img width="794" height="478" alt="Mask_screenshot_06 11 2025" src="https://github.com/user-attachments/assets/fdc50fe9-f5e3-42aa-9545-a315a95558c4" />

Use this tuner to make the fish appear white and the background black.
<img width="794" height="478" alt="Result_screenshot_06 11 2025" src="https://github.com/user-attachments/assets/f2e0fb66-a341-4995-9e42-68c441bb8ba8" />

#### 3. Contour Extraction and Bounding Box

```contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
```

<img width="794" height="478" alt="final-prototype" src="https://github.com/user-attachments/assets/c5ce0984-47d8-4150-8ed3-92f72c018f24" />

Red outline = fish contour
Green rectangle = bounding box
The detected contour accurately follows the fish boundary for further analysis.

#### 4. Measurement (Next Step)
Once the fish contour is detected, the next goal is to compute real-world measurements
Step 4.1: Calibrate Pixel Scale
Use a known object (e.g., a pipe or ruler in the image) to compute:
```
cm_per_pixel = known_length_cm / measured_length_px
```

#### Step 4.2: Compute Fish Dimensions
Convert from pixels to centimeters:

```length_cm = w * cm_per_pixel
height_cm = h * cm_per_pixel
area_cm2 = cv2.contourArea(largest_contour) * (cm_per_pixel ** 2)
```

#### Step 4.3: Estimate Weight (Future)

Using biological allometric scaling:
```
W=a×Lb
```
A regression model will be trained using known length–weight data from the farm to estimate fish weight.
```
Detected Fish Bounding Box: 141px width × 48px height
Contour area: 15844 px²
(After calibration)
Estimated Fish Length: 36.1 cm
Estimated Fish Area: 425.5 cm²
```








