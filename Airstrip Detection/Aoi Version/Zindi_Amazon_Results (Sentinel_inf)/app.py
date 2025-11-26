import os
import uuid
import shutil
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# --- START: Recommended Path Fix ---
# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Import your pipeline functions
from detect_airstrips_pipeline import (
    split_aoi_into_tiles,
    process_tiles_with_yolo,
    apply_nms,
    visualize_detections_on_aoi
)

# Folders and configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
# Ensure absolute path is used for the model, or use a path relative to BASE_DIR
YOLO_MODEL_PATH = r'Zindi_Amazon_Results (Sentinel_inf)\best.pt'
# --- END: Recommended Path Fix ---


ALLOWED_EXTENSIONS = {"tif", "tiff"}

# Use the absolute paths defined above
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
# Limit uploads to 512MB (adjust as needed)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    result_url = None

    if request.method == "POST":
        if "file" not in request.files:
            error_message = "No file part in the request."
            return render_template("index.html", error_message=error_message, result_url=result_url)

        file = request.files["file"]
        if file.filename == "":
            error_message = "No file selected. Please choose a GeoTIFF (.tif or .tiff)."
            return render_template("index.html", error_message=error_message, result_url=result_url)

        if not allowed_file(file.filename):
            error_message = "Invalid file type. Only .tif and .tiff files are allowed."
            return render_template("index.html", error_message=error_message, result_url=result_url)

        # Generate unique IDs and safe paths
        output_id = uuid.uuid4().hex[:8]
        safe_name = secure_filename(file.filename)
        ext = os.path.splitext(safe_name)[1].lower()

        tif_name = f"upload_{output_id}{ext}"
        tif_path = os.path.join(app.config["UPLOAD_FOLDER"], tif_name)

        tiles_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"tiles_{output_id}")
        os.makedirs(tiles_dir, exist_ok=True)

        result_filename = f"result_{output_id}.png"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], result_filename)

        try:
            # Save upload
            file.save(tif_path)

            # Sanity check for model path
            if not os.path.exists(YOLO_MODEL_PATH):
                raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")

            # Run pipeline
            split_aoi_into_tiles(tif_path, tiles_dir, tile_size=512, overlap=256)
            detections = process_tiles_with_yolo(tiles_dir, YOLO_MODEL_PATH)
            detections = apply_nms(detections, iou_threshold=0.3)
            visualize_detections_on_aoi(tif_path, detections, output_path)

            # --- OPTIONAL: Debug Check (Helps confirm the file exists before serving) ---
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"The result image was not created by the pipeline at {output_path}")
            # --------------------------------------------------------------------------

            # Use the dedicated outputs route for serving results
            result_url = url_for("outputs", filename=result_filename)
            return render_template("index.html", result_url=result_url, error_message=None)

        except Exception as e:
            app.logger.exception("Processing failed")
            error_message = f"Processing failed: {str(e)}"
            return render_template("index.html", error_message=error_message, result_url=None)
        finally:
            # Cleanup intermediate files/dirs (keep only the output image)
            try:
                if os.path.isdir(tiles_dir):
                    shutil.rmtree(tiles_dir, ignore_errors=True)
            except Exception:
                app.logger.warning("Failed to clean up tiles directory.")
            try:
                if os.path.isfile(tif_path):
                    os.remove(tif_path)
            except Exception:
                app.logger.warning("Failed to remove uploaded file.")

    return render_template("index.html", error_message=error_message, result_url=result_url)


@app.route("/outputs/<filename>")
def outputs(filename):
    # Serve generated output images from the absolute path defined in app.config
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    # For development; consider setting debug=False in production
    app.run(debug=True)