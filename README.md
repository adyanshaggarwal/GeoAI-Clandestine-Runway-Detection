# GeoAI – Clandestine Runway Detection

# Satellite-Based Detection of Hidden Airstrips in the Amazon Basin

This repository contains two major components:

1. Annotation Tool – A web-based annotator for creating binary runway masks.
2. YOLOv8-Based Detection Pipeline – Code for training and running a machine learning model to detect hidden airstrips using satellite imagery (Sentinel-1/2 RGB mosaics) as documented in the final report and presentation .

---

# Repository Structure

```
root/
│── annotator/           # Web UI for mask annotation
│── data/                # Input images or satellite tiles
│── generated_png/       # Auto-generated PNG binary masks
│── generated_tiffs/     # Auto-generated TIFF binary masks
│── models/              # Trained YOLOv8 weights
│── scripts/             # Training, inference, and preprocessing utilities
│── app.py               # Main Annotator application
│── requirements.txt
│── README.md
```

---

# 1. Setup Instructions

1.1. Install Dependencies

Create a virtual environment (recommended) and install requirements:

```bash
pip install -r requirements.txt
```

---

# 2. Annotation Tool Usage

The **Annotator** helps you create binary runway masks from satellite PNG images.
(Referenced in the workflow in the project report’s methodology section .)

---

2.1 Launch the Annotator

Run:

```bash
python app.py
```

Your CLI will display a URL (typically `http://127.0.0.1:5000`).
Open it in your browser.

---

2.2 Upload Image Folder

Inside the UI:

* Select *Upload Folder*
* Upload a folder containing `.png` files
  (Example naming: `pancreas_001_slice_XXX.png` or `*_rgb.png` depending on dataset)

> The tool expects a folder, not individual images.

---

2.3 Annotating Runways

1. Click **two endpoints** on the image → a runway line is drawn.
2. A **binary mask** appears next to the image.
3. Press **Download** to save the mask.

If the image contains *no runway*:

Simply click **Download** without marking any points.

---

2.4 Output Files

Masks are automatically saved to:

```
generated_png/     # PNG binary masks
generated_tiffs/   # TIFF binary masks
```

Prefix `0_*.png` → No runway present
Prefix `1_*.png` → Runway present

You can monitor downloads via **Right-Click → Inspect → Console** in the browser.

---

# 3. YOLOv8 Detection Pipeline (Overview)

This repository also supports training and inference of the model described in the project report.
The method uses:

* Sentinel-2 RGB tiles
* Binary masks converted to YOLO bounding boxes
* YOLOv8-small (11.2M params)
* DEM filtering to eliminate infeasible terrains (slope >2°)
* OSM overlay to remove road-like false positives
  (From methodology, results, and pipeline diagrams in the PDF/PPT .)

To train or run inference, refer to the scripts inside `scripts/`.

---

# 4. Running Inference (Optional)

After placing weights in `models/`, you can run:

```bash
python scripts/run_inference.py --image <path-to-image> --weights models/best.pt
```

Outputs include:

* Predicted bounding boxes
* Confidence values
* (Optional) DEM + OSM post-processed detections

---

# 5. Dataset Notes

As described in the project report’s *Methodology & Data* section :

* 708 training images
* 177 validation images
* 307 withheld test images
* Binary runway masks → converted to YOLO format
* Sentinel-2 RGB input (10 m resolution)
* DEM data used for terrain filtering
* OSM used for false-positive suppression

---

# 6. Annotation → Training Flow

```
Raw PNG tiles  
    ↓  
Annotator → Binary mask  
    ↓  
scripts/convert_masks_to_yolo.py  
    ↓  
YOLO-formatted labels  
    ↓  
YOLOv8 Training  
    ↓  
DEM filtering + OSM overlay  
    ↓  
Final detections
```

(Full workflow explained in Chapters 4 & 5 of the report  and pipeline diagrams in the PPT .)

---

# 7. Output Structure Summary

| Folder             | Description                                   |
| ------------------ | --------------------------------------------- |
| `generated_png/`   | Annotated PNG binary masks                    |
| `generated_tiffs/` | Annotated TIFF masks                          |
| `labels/`          | YOLO-format bounding boxes (after conversion) |
| `runs/detect/`     | YOLO inference results                        |
| `models/`          | Trained `.pt` weights                         |

---

# 8. Credits

This work was carried out as part of the GeoAI: Amazon Basin Secret Runway Detection project by PES University & Nokia (2025), documented fully in:
Final report and Final PPT

Authors:
Kodandram Ranganath, Rajasekhar Mohan, Adyansh Aggarwal, Ishita Dalela, S. P. Suhas, Khushi Jayaprakash

---

