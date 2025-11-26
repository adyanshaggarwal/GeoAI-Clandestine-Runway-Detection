import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from rasterio.windows import Window
from PIL import Image
import os
import json

# --- Helper Functions ---

def convert_tif_to_png(tif_path, png_path, src):
    """Converts a TIF file to a PNG for YOLO prediction."""
    b2 = src.read(1, boundless=True, fill_value=0)
    b3 = src.read(2, boundless=True, fill_value=0)
    b4 = src.read(3, boundless=True, fill_value=0)

    rgb = np.stack([b4, b3, b2], axis=-1)
    rgb = np.nan_to_num(rgb)

    # Stretch to 0-255 safely
    rgb = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.uint8)
    img = Image.fromarray(rgb, mode='RGB')
    img.save(png_path)

# --- Project Pipeline Functions ---

def split_aoi_into_tiles(input_tif, output_dir, tile_size, overlap):
    """
    Splits a large GeoTIFF into smaller, overlapping tiles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(input_tif) as src:
        width = src.width
        height = src.height

        # Calculate stride with overlap
        stride_x = tile_size - overlap
        stride_y = tile_size - overlap

        # Iterate over the image and create tiles
        for i in range(0, height, stride_y):
            for j in range(0, width, stride_x):
                window = Window(j, i, tile_size, tile_size).intersection(Window(0, 0, width, height))

                # Create a new transform for the tile
                tile_transform = src.window_transform(window)

                # Create output filename
                tile_filename = os.path.join(output_dir, f"tile_{i}_{j}.tif")

                # Write the tile to a new GeoTIFF file
                with rasterio.open(
                    tile_filename,
                    'w',
                    driver='GTiff',
                    height=window.height,
                    width=window.width,
                    count=src.count,
                    dtype=src.read(1).dtype,
                    crs=src.crs,
                    transform=tile_transform
                ) as dst:
                    dst.write(src.read(window=window))
        print(f"✅ Split {input_tif} into tiles and saved to {output_dir}")

def process_tiles_with_yolo(tiles_dir, yolo_model_path, min_conf=0.30):
    """
    Processes each tile with a YOLO model and collects georeferenced bounding boxes.
    """
    yolo_model = YOLO(yolo_model_path)
    all_detections = []

    for filename in os.listdir(tiles_dir):
        if filename.endswith(".tif"):
            tif_path = os.path.join(tiles_dir, filename)
            png_path = tif_path.replace('.tif', '.png')

            with rasterio.open(tif_path) as src:
                # Convert TIF to PNG for YOLO model
                convert_tif_to_png(tif_path, png_path, src)

                # Run YOLO prediction
                results = yolo_model(png_path, conf=min_conf, verbose=False)

                # Extract and georeference bounding boxes
                transform = src.transform
                for r in results:
                    for box in r.boxes:
                        # Convert pixel coordinates to georeferenced coordinates
                        x1_px, y1_px, x2_px, y2_px = box.xyxy[0].tolist()
                        lon1, lat1 = transform * (x1_px, y1_px)
                        lon2, lat2 = transform * (x2_px, y2_px)

                        detection = {
                            "bbox": [lon1, lat1, lon2, lat2],
                            "confidence": box.conf.item(),
                            "class": r.names[box.cls.item()],
                            "pixel_coords": [x1_px, y1_px, x2_px, y2_px] # Store pixel coords for NMS
                        }
                        all_detections.append(detection)

            os.remove(png_path)
    print("✅ Processed all tiles and collected detections.")
    return all_detections

def apply_nms(detections, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to a list of detections.
    """
    if not detections:
        return []

    # Prepare data for NMS
    boxes = np.array([d['pixel_coords'] for d in detections])
    confidences = np.array([d['confidence'] for d in detections])
    
    # Calculate areas
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence
    order = confidences.argsort()[::-1]
    
    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)
        
        # Calculate Intersection over Union (IoU)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with low IoU
        indices_to_keep = np.where(iou <= iou_threshold)[0]
        order = order[indices_to_keep + 1]
    
    return [detections[i] for i in keep_indices]


def visualize_detections_on_aoi(original_tif, detections, output_png):
    """
    Draws georeferenced bounding boxes and confidence scores on the original AOI image.
    """
    with rasterio.open(original_tif) as src:
        img_array = src.read([3, 2, 1])
        transform = src.transform

        # Normalize and convert to 8-bit for visualization
        img_array = np.moveaxis(img_array, 0, -1)
        img_array = (255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Iterate through all detections and draw the boxes
        for detection in detections:
            lon1, lat1, lon2, lat2 = detection["bbox"]
            confidence = detection["confidence"]

            # Convert georeferenced coordinates back to pixel coordinates
            col1, row1 = ~transform * (lon1, lat1)
            col2, row2 = ~transform * (lon2, lat2)
            
            p1 = (int(col1), int(row1))
            p2 = (int(col2), int(row2))

            color = (0, 0, 255)
            thickness = 2
            
            # Draw the bounding box
            cv2.rectangle(img_bgr, p1, p2, color, thickness)
            
            # Prepare and draw the confidence text
            text = f"Confidence: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Position the text slightly above the top-left corner
            text_x = p1[0]
            text_y = p1[1] - 5 
            
            # Add a background rectangle for better readability
            cv2.rectangle(img_bgr, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)
            cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            
        cv2.imwrite(output_png, img_bgr)
        print(f"✅ Saved final image with detections to {output_png}")

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- User-Defined Paths and Parameters ---
    input_large_aoi_tif = "aoi\Sentinel_AllBands_Inference_2020_02.tif"
    output_tiles_dir = "output"
    yolo_model_path = "best.pt"
    final_output_image = "final_output_with_detections.png"

    # --- Pipeline Parameters ---
    tile_size = 512
    overlap = 256

    # --- Step 1: Split the large GeoTIFF into smaller tiles ---
    split_aoi_into_tiles(input_large_aoi_tif, output_tiles_dir, tile_size, overlap)

    # --- Step 2: Process each tile with YOLO and get detections ---
    detections = process_tiles_with_yolo(output_tiles_dir, yolo_model_path)
    
    # --- Step 2.5: Apply NMS to remove overlapping boxes ---
    detections = apply_nms(detections, iou_threshold=0.3)
    
    # --- Step 3: Visualize all detections on the original AOI ---
    visualize_detections_on_aoi(input_large_aoi_tif, detections, final_output_image)

    # --- Optional: Clean up temporary tiles ---
    # import shutil
    # shutil.rmtree(output_tiles_dir)
    # print(f"✅ Cleaned up temporary directory: {output_tiles_dir}")