````markdown name=PRODUCTIONGUIDE.md
# 📖 JuaSamaki Production Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Calibration](#calibration)
5. [Model Training](#model-training)
6. [Inference](#inference)
7. [Batch Processing](#batch-processing)
8. [Python API](#python-api)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/KaraniWachira/juasamaki.git
cd juasamaki

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note on Dependencies:**
- `opencv-python`: Computer vision
- `numpy`, `pandas`: Data processing
- `scikit-learn`: Machine learning
- `matplotlib`: Visualization
- `pyyaml`: Configuration management

---

## Quick Start

### 1️⃣ Create Calibration Profile

First, you need to calibrate your camera. You'll need an object of known size (e.g., a 10cm ruler).

```bash
python main.py sample_image.jpg --cal-length-cm 10.0 --cal-length-px 95.0 --calibration camera1
```

This creates a calibration profile named "camera1" (saved in `calibrations/camera1.json`).

### 2️⃣ Process Single Image

```bash
python main.py image.jpg --calibration camera1 --hsv-preset bright --visualize
```

This will:
- Detect fish using HSV color segmentation
- Measure fish dimensions (length, height, area)
- Save annotated image showing detected fish

### 3️⃣ Process Multiple Images

```bash
python main.py /path/to/images/ --calibration camera1 --visualize --export-csv results.csv
```

This will:
- Process all .jpg/.png files in the directory
- Save detection visualizations
- Export all measurements to CSV

### 4️⃣ Train Weight Prediction Model

```bash
python train_model.py farm_measurements.csv --model-type random_forest --output models/catfish.pkl --plot analysis/
```

**Expected CSV format:**
```
length_cm,height_cm,area_cm2,weight_kg
30.5,8.2,420.1,0.85
32.1,8.5,450.3,0.92
...
```

### 5️⃣ Complete Analysis with Weight Prediction

```bash
python main.py /daily_harvest/ \
    --calibration camera1 \
    --model models/catfish.pkl \
    --visualize \
    --export-csv results.csv \
    --report harvest_report.html \
    --plot
```

---

## Configuration

### HSV Presets

Available presets for different lighting conditions:

```bash
# List all presets
python main.py image.jpg --hsv-preset bright --list-presets

# Available: bright, low_light, underwater, mixed, catfish_farm
```

**Default Presets:**

| Preset | Best For | HSV Bounds |
|--------|----------|-----------|
| `bright` | Sunny outdoor conditions | (0,30,60) → (20,255,255) |
| `low_light` | Dim/shadowy conditions | (0,20,40) → (25,255,255) |
| `underwater` | Blue-tinted/underwater | (85,30,60) → (130,255,255) |
| `mixed` | Variable lighting | (0,25,50) → (30,255,255) |
| `catfish_farm` | Catfish-optimized | (0,30,60) → (20,255,255) |

### Create Custom Preset

In Python:
```python
from src.config import ConfigManager

config = ConfigManager()
config.create_custom_preset(
    name="my_preset",
    lower=(0, 25, 50),
    upper=(25, 255, 255),
    description="Custom lighting conditions"
)
```

Or edit `configs/presets.yaml`:
```yaml
my_preset:
  lower: [0, 25, 50]
  upper: [25, 255, 255]
  description: Custom lighting conditions
```

---

## Calibration

### Understanding Calibration

Calibration converts pixel measurements to real-world dimensions (centimeters).

**What You Need:**
- An object of known length (ruler, pipe, reference object)
- Image with the object clearly visible

**Process:**
1. Measure object length in cm (e.g., 10 cm)
2. Measure object length in pixels in the image (e.g., 95 pixels)
3. Create calibration: 10 cm ÷ 95 px = 0.1053 cm/px

### Create Calibration

**Method 1: From command line**
```bash
python main.py reference_image.jpg \
    --cal-length-cm 10.0 \
    --cal-length-px 95.0 \
    --calibration camera1
```

**Method 2: In Python**
```python
from src.calibration import CalibrationManager

cal_manager = CalibrationManager()
cal = cal_manager.create_calibration(
    name="camera1",
    reference_length_cm=10.0,
    reference_length_px=95.0,
    camera_id="Phone Camera",
    notes="Calibrated on 2026-05-05"
)
```

### View Calibrations

```bash
python main.py image.jpg --list-calibrations
```

Output:
```
📏 Available Calibrations
=================================================================
camera1
  Camera: Phone Camera
  Reference: 10.0 cm = 95.0 px
  Factor: 0.105263 cm/px
  Created: 2026-05-05T10:30:45
  Notes: Calibrated on 2026-05-05
```

### Manage Calibrations

```python
from src.calibration import CalibrationManager

cal_manager = CalibrationManager()

# List all
cal_manager.list_calibrations()

# Get specific
cal = cal_manager.get_calibration("camera1")
print(f"Factor: {cal.cm_per_pixel}")

# Export/Import
cal_manager.export_calibrations("backup.json")
cal_manager.import_calibrations("backup.json")

# Delete
cal_manager.delete_calibration("camera1")
```

---

## Model Training

### Dataset Preparation

Create a CSV with ground truth measurements and weights:

```csv
length_cm,height_cm,area_cm2,weight_kg
25.3,7.1,310.2,0.65
28.5,7.8,365.4,0.78
30.1,8.2,420.5,0.92
32.4,8.9,485.3,1.15
...
```

**Column Requirements:**
- `length_cm`: Fish length (calibrated)
- `height_cm`: Fish height (calibrated)
- `area_cm2`: Fish contour area (calibrated)
- `weight_kg`: Actual weight (ground truth)

**Data Collection Tips:**
1. Measure at least 50 fish for reliable model
2. Ensure measurements are accurate
3. Include diverse sizes (small to large)
4. Use same calibration profile for all images

### Train Model

```bash
# Random Forest (recommended)
python train_model.py farm_data.csv \
    --model-type random_forest \
    --output models/catfish.pkl \
    --plot analysis/

# Linear Regression (simpler)
python train_model.py farm_data.csv \
    --model-type linear \
    --output models/catfish_linear.pkl

# With custom cross-validation
python train_model.py farm_data.csv \
    --model-type random_forest \
    --test-size 0.25 \
    --cv-folds 10 \
    --output models/catfish.pkl
```

### Interpret Results

The trainer outputs:
```
============================================================
🤖 Model Training Complete (random_forest)
============================================================
Test R² Score: 0.9234
Test RMSE: 0.0847 kg
Test MAE: 0.0612 kg
Cross-validation R² (mean±std): 0.9156 ± 0.0234
============================================================
```

**Metrics Explanation:**
- **R² Score**: Higher is better (0-1). 0.92 = 92% variance explained
- **RMSE**: Root Mean Squared Error. 0.085 kg = average error
- **MAE**: Mean Absolute Error. 0.061 kg = typical error magnitude
- **CV R²**: Cross-validation score (robustness check)

**Good Model:** R² > 0.85, MAE < 10% of average weight

### Analyze Model

The trainer generates plots:
1. **Feature Distributions** - How features are spread
2. **Weight Distribution** - Target variable distribution
3. **Feature Importance** - Which features matter most

---

## Inference

### Single Image

```bash
python main.py fish_image.jpg \
    --calibration camera1 \
    --model models/catfish.pkl \
    --visualize
```

Output:
```
✅ Detected 3 fish

  Fish #1:
    Length: 28.50 cm
    Height: 7.80 cm
    Weight: 0.78 kg

  Fish #2:
    Length: 31.20 cm
    Height: 8.50 cm
    Weight: 0.95 kg

  Fish #3:
    Length: 29.80 cm
    Height: 8.10 cm
    Weight: 0.87 kg
```

### Different Lighting

```bash
# Bright outdoor
python main.py image.jpg --calibration camera1 --hsv-preset bright

# Low light
python main.py image.jpg --calibration camera1 --hsv-preset low_light

# Underwater
python main.py image.jpg --calibration camera1 --hsv-preset underwater
```

---

## Batch Processing

### Process Directory

```bash
python main.py /farm/harvest_2026_05_05/ \
    --calibration camera1 \
    --model models/catfish.pkl \
    --visualize \
    --export-csv results.csv \
    --report harvest_report.html \
    --plot
```

**Output Structure:**
```
/farm/harvest_2026_05_05/
├── image1.jpg → detected → image1_detected.jpg
├── image2.jpg → detected → image2_detected.jpg
├── results.csv (all measurements)
├── harvest_report.html (summary report)
└── analysis/
    ├── 01_feature_distributions.png
    ├── 02_weight_distribution.png
    ├── 03_length_vs_weight.png
    └── 04_dimension_summary.png
```

### CSV Output Format

```csv
image,fish_id,length_cm,height_cm,area_cm2,predicted_weight_kg
image1.jpg,1,28.50,7.80,310.2,0.78
image1.jpg,2,31.20,8.50,365.4,0.95
image2.jpg,1,29.80,8.10,320.5,0.87
...
```

### Generate Reports

```bash
python main.py /images/ \
    --calibration camera1 \
    --report summary.html \
    --export-csv measurements.csv
```

This creates:
- **summary.html**: Interactive HTML report with statistics
- **measurements.csv**: All raw data for spreadsheet analysis

---

## Python API

### Basic Usage

```python
from src.fish_detector import FishDetector
from src.calibration import CalibrationManager
from src.weight_predictor import WeightPredictor
from src.pipeline import FishAnalysisPipeline

# Initialize
detector = FishDetector((0, 30, 60), (20, 255, 255))
cal_manager = CalibrationManager()
calibration = cal_manager.get_calibration("camera1")

# Detect
result = detector.detect("image.jpg")
detector.print_detection_summary(result)

# Measure
for fish in result.fish_contours:
    measurements = detector.get_measurements(fish, calibration.cm_per_pixel)
    print(f"Length: {measurements['length_cm']:.2f} cm")

# Visualize
vis_image = detector.visualize_detection("image.jpg", result, "output.jpg")
```

### With Weight Prediction

```python
# Load model
predictor = WeightPredictor()
predictor.load_model("models/catfish.pkl")

# Predict
for fish in result.fish_contours:
    m = detector.get_measurements(fish, calibration.cm_per_pixel)
    weight, features = predictor.predict(
        m["length_cm"],
        m["height_cm"],
        m["area_cm2"]
    )
    print(f"Predicted weight: {weight:.2f} kg")
```

### Complete Pipeline

```python
from src.pipeline import FishAnalysisPipeline

pipeline = FishAnalysisPipeline(detector, calibration, predictor)

# Single image
result = pipeline.process_image("image.jpg", save_visualization=True)

# Multiple images
measurements = pipeline.process_directory(
    "/path/to/images/",
    save_visualizations=True
)

# Get statistics
pipeline.print_summary()

# Export
pipeline.export_to_csv("results.csv")
```

### Training

```python
from src.weight_predictor import WeightPredictor, FishMeasurement

# Load data
measurements = [
    FishMeasurement(25.3, 7.1, 310.2, 0.65),
    FishMeasurement(28.5, 7.8, 365.4, 0.78),
    # ... more measurements
]

# Train
predictor = WeightPredictor(model_type="random_forest")
metrics = predictor.train(measurements)

print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f} kg")

# Save
predictor.save_model("models/catfish.pkl")
```

---

## Troubleshooting

### Issue: No Fish Detected

**Symptoms:** "Detected 0 fish" message

**Causes & Solutions:**

1. **Wrong HSV preset for lighting**
   ```bash
   # Try different preset
   python main.py image.jpg --hsv-preset low_light
   ```

2. **HSV bounds don't match fish color**
   ```python
   # Adjust HSV bounds
   detector = FishDetector((0, 20, 40), (25, 255, 255))
   ```

3. **Image quality too low**
   - Ensure good lighting, sharp focus
   - Use higher resolution camera

### Issue: Inaccurate Measurements

**Symptoms:** Measured sizes don't match visual assessment

**Causes & Solutions:**

1. **Calibration error**
   ```bash
   # Recalibrate with more accurate reference
   python main.py ref_image.jpg --cal-length-cm 10.0 --cal-length-px 95.0 --calibration camera1
   ```

2. **Fish not fully segmented**
   - Adjust HSV bounds
   - Check image contrast
   - Ensure fish is fully visible

3. **Partial fish detection**
   - Increase `min_area` parameter
   - Check for debris mistaken as fish

### Issue: Poor Weight Predictions

**Symptoms:** Predicted weights seem off

**Causes & Solutions:**

1. **Insufficient training data**
   - Collect at least 100 fish measurements
   - Ensure diverse sizes

2. **Model not loaded**
   ```bash
   python main.py image.jpg --model models/catfish.pkl
   ```

3. **Model trained on different conditions**
   - Retrain model with data from current farm
   - Use consistent calibration and lighting

### Issue: File Not Found Error

```
❌ Could not load image: [Errno 2] No such file or directory
```

**Solution:**
```bash
# Use absolute path
python main.py /full/path/to/image.jpg --calibration camera1

# Or run from correct directory
cd /path/with/images
python ../../main.py image.jpg --calibration camera1
```

### Issue: Calibration Not Found

```
❌ Calibration not found. Use --calibration or provide...
```

**Solution:**
```bash
# List available calibrations
python main.py image.jpg --list-calibrations

# Create new calibration
python main.py image.jpg --cal-length-cm 10 --cal-length-px 95 --calibration mycam
```

### Issue: Model Load Error

```
❌ Error loading model
```

**Solution:**
```bash
# Verify model file exists
ls models/

# Retrain if needed
python train_model.py data.csv --model-type random_forest --output models/catfish.pkl
```

---

## Performance Tips

### Speed Optimization

1. **Reduce image resolution** (if acceptable for your use case)
2. **Use simpler HSV preset** (fewer morphological operations)
3. **Process in parallel** (when batch processing)

### Accuracy Optimization

1. **Use high-quality calibration** (measure reference precisely)
2. **Train with > 100 fish** (more data = better model)
3. **Include diverse sizes** (small to large fish)
4. **Use Random Forest** over Linear Regression (better accuracy)

---

## Support & Contact

For issues, questions, or contributions:
- GitHub Issues: https://github.com/KaraniWachira/juasamaki/issues
- Email: wachirakeith@gmail.com

---

**Happy fishing! 🎣**
````
