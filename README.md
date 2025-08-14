# Amazon Runway Detector

A YOLO-powered web application for detecting airstrips in Amazon rainforest satellite imagery with real-time processing and beautiful visualizations.

## Features

- **🎯 Advanced YOLO Detection**: Uses your trained YOLO model to identify runways in satellite images
- **🎨 Beautiful Modern UI**: Vibrant, responsive interface with Amazon rainforest theme
- **⚡ Real-time Processing**: Upload images and get instant detection results with live feedback
- **📊 Detailed Analytics**: Confidence scores, position data, and comprehensive summary statistics
- **💾 Download Results**: Save annotated images with detection overlays and bounding boxes
- **🔄 Smart Fallback**: Automatic demo mode when model is unavailable

## Quick Start

### 1. Install Dependencies

\`\`\`bash
# Install Node.js dependencies
npm install

# Install Python dependencies for YOLO
pip install ultralytics opencv-python pillow numpy
\`\`\`

### 2. Add Your YOLO Model

Place your trained YOLO model file `best.pt` in the root directory:

\`\`\`
project-root/
├── best.pt          # <-- Place your YOLO model here
├── app/
├── components/
└── ...
\`\`\`

**Note**: Without the model, the app runs in demo mode with sample data.

### 3. Run the Application

\`\`\`bash
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) to start detecting runways!

## How It Works

1. **Upload**: Select a satellite image of the Amazon rainforest
2. **Process**: The system automatically detects if your YOLO model is available
3. **Analyze**: Real YOLO detection or demo mode with clear status indicators
4. **Results**: View detections with confidence scores, positions, and dimensions
5. **Download**: Save annotated images with professional bounding box overlays

## Detection Modes

The application intelligently switches between three modes:

- **🟢 Production Mode**: Uses your `best.pt` model for real detection
- **🔵 Demo Mode**: Shows sample results when model is unavailable  
- **🟡 Error Fallback**: Graceful fallback if model encounters issues

## API Reference

### POST /api/detect

Upload an image for runway detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field (max 10MB)

**Response:**
\`\`\`json
{
  "detections": [
    {
      "label": "Airstrip 1",
      "confidence": 86.6,
      "position": { "x": 82, "y": 36 },
      "dimensions": { "width": 293, "height": 57 }
    }
  ],
  "summary": {
    "total_detections": 3,
    "average_confidence": 89.0,
    "max_confidence": 95.8,
    "image_resolution": { "width": 513, "height": 514 }
  },
  "annotated_image": "<base64_image_string>",
  "mode": "production"
}
\`\`\`

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes with Python integration
- **UI**: shadcn/ui components with custom Amazon theme
- **AI**: YOLO (You Only Look Once) object detection
- **Styling**: Gradient backgrounds, glassmorphism effects, responsive design

## Project Structure

\`\`\`
├── app/
│   ├── api/detect/route.ts    # Smart YOLO detection API
│   ├── page.tsx               # Main application with enhanced UI
│   ├── layout.tsx             # Root layout
│   └── globals.css            # Global styles
├── components/ui/             # shadcn/ui components
├── best.pt                    # Your YOLO model (place here)
└── README.md
\`\`\`

## Troubleshooting

### Common Issues

**🔴 "Demo mode" showing instead of real detection**
- Ensure `best.pt` is in the root directory
- Check file permissions and model file integrity

**🔴 Python/YOLO errors**
- Install dependencies: `pip install ultralytics opencv-python pillow numpy`
- Verify Python 3.8+ is installed
- Check console logs for detailed error messages

**🔴 Upload failures**
- File must be under 10MB
- Supported formats: JPG, PNG, GIF, WebP
- Check network connection for large files

**🔴 Slow processing**
- Large images take longer to process
- Consider resizing images to 1024x1024 or smaller
- Ensure sufficient system memory for YOLO model

### Performance Tips

- **Optimal image size**: 512x512 to 1024x1024 pixels
- **Best formats**: JPG for photos, PNG for screenshots
- **Model placement**: Keep `best.pt` in root directory for fastest access

## Development

The application includes comprehensive error handling, loading states, and user feedback. The UI automatically adapts based on detection results and provides clear status indicators for different operational modes.

For development without a YOLO model, the app provides realistic demo data to test all UI components and workflows.
