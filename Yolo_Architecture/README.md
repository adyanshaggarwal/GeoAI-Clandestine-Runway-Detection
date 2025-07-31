# 🛩️ Airstrip Detection with YOLOv8

This project prepares and trains a YOLOv8 model to detect airstrips from satellite images using binary mask annotations.

---

## 📦 1. `convert_to_yolo_bounds_test_1.ipynb`

**Purpose:**  
Converts binary mask images into YOLO-formatted bounding box annotations.

**Steps:**
- Loads PNG masks and matches them with corresponding satellite image chips.
- Extracts bounding boxes from non-zero mask areas.
- Adds optional padding to each box.
- Saves:
  - Processed images (`.png`)
  - YOLO labels (`.txt`)
  - Empty `.txt` for images with no objects (false samples)
- Output is structured for YOLO training.

---

## 🏋️ 2. `Yolo2.ipynb`

**Purpose:**  
Trains a YOLOv8 model on the dataset created above.

**Features:**
- Splits images into train/val (80/20)
- Creates `dataset.yaml` automatically
- Trains YOLOv8 with GPU support (if available)
- Saves each model to Google Drive with a timestamped folder
- Saves metrics (mAP, precision, recall) to `metrics.txt`
- Includes code to test predictions on custom uploaded images
- Visualizes top prediction with bounding box overlay

---

## 📁 Folder Structure After Conversion

```
yolo_airstrip_dataset/
├── images/
│   ├── train/
│   ├── train_split/
│   └── val/
├── labels/
│   ├── train/
│   ├── train_split/
│   └── val/
├── dataset.yaml
```

---

## Requirements

- Google Colab
- Python 3
- Packages:
  - `ultralytics`
  - `opencv-python-headless`
  - `rasterio`
  - `numpy`
  - `matplotlib`
  - `tqdm`