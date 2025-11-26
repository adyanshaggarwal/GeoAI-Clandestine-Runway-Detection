import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import fs from "fs"
import path from "path"

// Check if YOLO model exists
function hasYOLOModel(): boolean {
  const modelPath = path.join(process.cwd(), "best.pt")
  return fs.existsSync(modelPath)
}

// Mock YOLO detection function for development (when model is not available)
async function mockYOLODetection(imageBuffer: Buffer): Promise<any> {
  // Simulate processing time
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // Mock detection results
  const mockResults = {
    detections: [
      {
        label: "Airstrip 1",
        confidence: 86.6,
        position: { x: 82, y: 36 },
        dimensions: { width: 293, height: 57 },
      },
      {
        label: "Airstrip 2",
        confidence: 84.6,
        position: { x: 203, y: 86 },
        dimensions: { width: 173, height: 22 },
      },
      {
        label: "Airstrip 3",
        confidence: 95.8,
        position: { x: 43, y: 183 },
        dimensions: { width: 160, height: 64 },
      },
    ],
    summary: {
      total_detections: 3,
      average_confidence: 89.0,
      max_confidence: 95.8,
      image_resolution: { width: 513, height: 514 },
    },
    annotated_image:
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
    mode: "mock",
  }

  return mockResults
}

async function runYOLODetection(imageBuffer: Buffer): Promise<any> {
  try {
    // Check if model exists
    if (!hasYOLOModel()) {
      console.log("YOLO model (best.pt) not found, using mock data")
      const mockResult = await mockYOLODetection(imageBuffer)
      mockResult.mode = "mock"
      return mockResult
    }

    // Save uploaded image temporarily
    const tempImagePath = path.join(process.cwd(), "temp_image.jpg")
    fs.writeFileSync(tempImagePath, imageBuffer)

    const normalizedPath = tempImagePath.replace(/\\/g, "/")
    const modelPath = path.join(process.cwd(), "best.pt").replace(/\\/g, "/")

    const pythonScript = `
import sys
import json
import base64
import warnings
import os
import logging

# Suppress all warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['YOLO_VERBOSE'] = 'False'

try:
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Load the model with verbose=False
    model = YOLO(r"${modelPath}", verbose=False)
    
    # Run inference with verbose=False
    results = model(r"${normalizedPath}", verbose=False)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                # Only include detections with confidence > 0.5
                if conf > 0.5:
                    detections.append({
                        "label": f"Airstrip {len(detections)+1}",
                        "confidence": round(conf * 100, 1),
                        "position": {"x": int(x1), "y": int(y1)},
                        "dimensions": {"width": int(x2-x1), "height": int(y2-y1)}
                    })
    
    # Create annotated image
    img = Image.open(r"${normalizedPath}")
    draw = ImageDraw.Draw(img)
    
    # Draw bounding boxes and labels
    for detection in detections:
        x, y = detection["position"]["x"], detection["position"]["y"]
        w, h = detection["dimensions"]["width"], detection["dimensions"]["height"]
        
        # Draw bounding box
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        
        # Draw label background
        label = f"{detection['label']} ({detection['confidence']}%)"
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((x, y-25), label, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((x, y-25), label, fill="white", font=font)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    annotated_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Calculate summary
    total_detections = len(detections)
    avg_confidence = sum(d["confidence"] for d in detections) / total_detections if detections else 0
    max_confidence = max(d["confidence"] for d in detections) if detections else 0
    
    result = {
        "detections": detections,
        "summary": {
            "total_detections": total_detections,
            "average_confidence": round(avg_confidence, 1),
            "max_confidence": round(max_confidence, 1),
            "image_resolution": {"width": img.width, "height": img.height}
        },
        "annotated_image": annotated_base64,
        "mode": "production"
    }
    
    # Print only the JSON result
    print("JSON_START" + json.dumps(result) + "JSON_END")
    
except Exception as e:
    error_result = {
        "error": str(e),
        "detections": [],
        "summary": {
            "total_detections": 0,
            "average_confidence": 0,
            "max_confidence": 0,
            "image_resolution": {"width": 0, "height": 0}
        },
        "annotated_image": "",
        "mode": "error"
    }
    print("JSON_START" + json.dumps(error_result) + "JSON_END")
`

    return new Promise((resolve, reject) => {
      const pythonCommand = process.platform === "win32" ? "python" : "python3"
      const python = spawn(pythonCommand, ["-c", pythonScript])
      let output = ""
      let error = ""

      python.stdout.on("data", (data) => {
        output += data.toString()
      })

      python.stderr.on("data", (data) => {
        error += data.toString()
      })

      python.on("close", (code) => {
        // Clean up temp file
        try {
          fs.unlinkSync(tempImagePath)
        } catch (e) {
          console.log("Could not delete temp file:", e)
        }

        if (code === 0) {
          try {
            const jsonStart = output.indexOf("JSON_START")
            const jsonEnd = output.indexOf("JSON_END")

            if (jsonStart !== -1 && jsonEnd !== -1) {
              const jsonString = output.substring(jsonStart + 10, jsonEnd)
              const result = JSON.parse(jsonString)
              resolve(result)
            } else {
              console.error("No JSON markers found in output:")
              console.error("Raw output:", output)
              console.error("Error output:", error)
              reject(new Error("No valid JSON found in Python output"))
            }
          } catch (e) {
            console.error("Failed to parse YOLO output:")
            console.error("Raw output:", output)
            console.error("Error output:", error)
            console.error("Parse error:", e)
            reject(new Error("Failed to parse YOLO output"))
          }
        } else {
          console.error("Python script failed with code:", code)
          console.error("Error output:", error)
          console.error("Standard output:", output)
          reject(new Error(`YOLO detection failed: ${error}`))
        }
      })
    })
  } catch (error) {
    console.error("YOLO detection error:", error)
    // Fallback to mock data on error
    const mockResult = await mockYOLODetection(imageBuffer)
    mockResult.mode = "error_fallback"
    return mockResult
  }
}

async function runBulkYOLODetection(imageBuffers: { buffer: Buffer; filename: string }[]): Promise<any> {
  const allResults = []
  let totalDetections = 0
  let totalConfidenceSum = 0
  let maxConfidence = 0
  const totalImages = imageBuffers.length

  for (const { buffer, filename } of imageBuffers) {
    try {
      const result = await runYOLODetection(buffer)

      // Add filename to result
      result.filename = filename
      allResults.push(result)

      // Accumulate statistics
      totalDetections += result.summary.total_detections
      totalConfidenceSum += result.summary.average_confidence * result.summary.total_detections
      maxConfidence = Math.max(maxConfidence, result.summary.max_confidence)
    } catch (error) {
      console.error(`Error processing ${filename}:`, error)
      // Add error result for this image
      allResults.push({
        filename,
        error: error instanceof Error ? error.message : "Processing failed",
        detections: [],
        summary: { total_detections: 0, average_confidence: 0, max_confidence: 0 },
        mode: "error",
      })
    }
  }

  // Calculate combined statistics
  const overallAverageConfidence = totalDetections > 0 ? totalConfidenceSum / totalDetections : 0

  return {
    results: allResults,
    combined_summary: {
      total_images: totalImages,
      total_detections: totalDetections,
      average_confidence: Math.round(overallAverageConfidence * 10) / 10,
      max_confidence: maxConfidence,
      images_with_detections: allResults.filter((r) => r.summary?.total_detections > 0).length,
    },
    mode: allResults[0]?.mode || "bulk",
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    const files = formData.getAll("images") as File[]
    const singleFile = formData.get("image") as File

    if (files.length > 0) {
      // Bulk processing
      if (files.length > 50) {
        return NextResponse.json({ error: "Maximum 50 images allowed per batch" }, { status: 400 })
      }

      const imageBuffers = []
      for (const file of files) {
        // Validate file type
        if (!file.type.startsWith("image/")) {
          return NextResponse.json({ error: `File ${file.name} must be an image` }, { status: 400 })
        }

        // Validate file size (max 10MB per file)
        if (file.size > 10 * 1024 * 1024) {
          return NextResponse.json({ error: `File ${file.name} size must be less than 10MB` }, { status: 400 })
        }

        const bytes = await file.arrayBuffer()
        const buffer = Buffer.from(bytes)
        imageBuffers.push({ buffer, filename: file.name })
      }

      // Run bulk YOLO detection
      const results = await runBulkYOLODetection(imageBuffers)
      return NextResponse.json(results)
    } else if (singleFile) {
      // Single file processing (existing functionality)
      if (!singleFile.type.startsWith("image/")) {
        return NextResponse.json({ error: "File must be an image" }, { status: 400 })
      }

      if (singleFile.size > 10 * 1024 * 1024) {
        return NextResponse.json({ error: "File size must be less than 10MB" }, { status: 400 })
      }

      const bytes = await singleFile.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const results = await runYOLODetection(buffer)

      return NextResponse.json(results)
    } else {
      return NextResponse.json({ error: "No image files provided" }, { status: 400 })
    }
  } catch (error) {
    console.error("API error:", error)
    return NextResponse.json(
      {
        error: "Internal server error during image processing",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
