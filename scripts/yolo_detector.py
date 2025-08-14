"""
YOLO Runway Detection Script

This script demonstrates how to integrate the YOLO model for runway detection.
Place your best.pt file in the root directory of the backend.

Requirements:
- ultralytics
- Pillow
- numpy

Install with: pip install ultralytics Pillow numpy
"""

import sys
import json
import base64
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import os

def load_model(model_path="best.pt"):
    """Load the YOLO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please place your best.pt file in the root directory.")
    
    return YOLO(model_path)

def detect_runways(model, image_path):
    """Run runway detection on an image"""
    # Run inference
    results = model(image_path)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                detections.append({
                    "label": f"Airstrip {i+1}",
                    "confidence": round(conf * 100, 1),
                    "position": {"x": int(x1), "y": int(y1)},
                    "dimensions": {"width": int(x2-x1), "height": int(y2-y1)}
                })
    
    return detections, results

def create_annotated_image(image_path, detections):
    """Create annotated image with bounding boxes"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        x, y = detection["position"]["x"], detection["position"]["y"]
        w, h = detection["dimensions"]["width"], detection["dimensions"]["height"]
        
        # Draw bounding box
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        
        # Draw label with background
        label = f"{detection['label']} ({detection['confidence']}%)"
        bbox = draw.textbbox((x, y-25), label, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((x, y-25), label, fill="white", font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    annotated_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return annotated_base64, img.size

def main(image_path):
    """Main detection function"""
    try:
        # Load model
        model = load_model()
        
        # Run detection
        detections, results = detect_runways(model, image_path)
        
        # Create annotated image
        annotated_base64, image_size = create_annotated_image(image_path, detections)
        
        # Calculate summary statistics
        total_detections = len(detections)
        avg_confidence = sum(d["confidence"] for d in detections) / total_detections if detections else 0
        max_confidence = max(d["confidence"] for d in detections) if detections else 0
        
        # Prepare result
        result = {
            "detections": detections,
            "summary": {
                "total_detections": total_detections,
                "average_confidence": round(avg_confidence, 1),
                "max_confidence": round(max_confidence, 1),
                "image_resolution": {"width": image_size[0], "height": image_size[1]}
            },
            "annotated_image": annotated_base64
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yolo_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = main(image_path)
    print(json.dumps(result))
