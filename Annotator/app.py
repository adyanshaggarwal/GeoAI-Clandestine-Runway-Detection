# run requirements.txt
# Flask, OpenCV, NumPy, Pillow  
# This script sets up a Flask application to generate and save masks from images based on user input points.



# to run this do python app.py


from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__, template_folder='templates')

# Output folders
PNG_FOLDER = 'generated_png'
TIF_FOLDER = 'generated_tif'
os.makedirs(PNG_FOLDER, exist_ok=True)
os.makedirs(TIF_FOLDER, exist_ok=True)

def generate_mask_from_bytes(image_bytes, points, thickness=4):
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    mask = np.zeros_like(img)
    if len(points) == 2:
        pt1, pt2 = tuple(points[0]), tuple(points[1])
        cv2.line(mask, pt1, pt2, 255, thickness)
    return mask

@app.route('/')
def index():
    return render_template('mask_gen.html')

@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    file = request.files['image']
    points = eval(request.form['points'])
    mask = generate_mask_from_bytes(file.read(), points)
    if mask is None:
        return jsonify({'error': 'Could not generate mask'}), 400
    buffer = BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'mask_image': f"data:image/png;base64,{encoded}"})

@app.route('/save_mask', methods=['POST'])
def save_mask():
    file = request.files['image']
    points = eval(request.form['points'])  # could be []
    original_filename = request.form['filename']

    has_runway = len(points) == 2
    prefix = '1_' if has_runway else '0_'
    base_name = os.path.splitext(original_filename)[0]
    save_name = prefix + base_name

    if has_runway:
        mask = generate_mask_from_bytes(file.read(), points)
    else:
        img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        mask = np.zeros_like(img)

    png_path = os.path.join(PNG_FOLDER, f"{save_name}.png")
    tif_path = os.path.join(TIF_FOLDER, f"{save_name}.tif")

    cv2.imwrite(png_path, mask)
    Image.fromarray(mask).save(tif_path, format='TIFF')

    return jsonify({'success': True, 'paths': {'png': png_path, 'tif': tif_path}})

if __name__ == '__main__':
    app.run(debug=True)
