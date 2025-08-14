"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  Download,
  Zap,
  MapPin,
  Target,
  Activity,
  AlertTriangle,
  CheckCircle,
  Info,
  Satellite,
  Radar,
  Eye,
  Sparkles,
  FolderOpen,
  Files,
  ArrowRight,
} from "lucide-react"

interface Detection {
  label: string
  confidence: number
  position: { x: number; y: number }
  dimensions: { width: number; height: number }
}

interface DetectionResult {
  detections: Detection[]
  summary: {
    total_detections: number
    average_confidence: number
    max_confidence: number
    image_resolution: { width: number; height: number }
  }
  annotated_image: string
  mode?: string
  filename?: string
}

interface BulkDetectionResult {
  results: DetectionResult[]
  combined_summary: {
    total_images: number
    total_detections: number
    average_confidence: number
    max_confidence: number
    images_with_detections: number
  }
  mode: string
}

export default function YOLORunwayDetector() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [uploadMode, setUploadMode] = useState<"single" | "bulk">("single")
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<DetectionResult | null>(null)
  const [bulkResults, setBulkResults] = useState<BulkDetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setSelectedFiles(null)
      setUploadMode("single")
      setResults(null)
      setBulkResults(null)
      setError(null)

      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (files && files.length > 0) {
      setSelectedFiles(files)
      setSelectedFile(null)
      setUploadMode("bulk")
      setResults(null)
      setBulkResults(null)
      setError(null)
      setPreviewUrl(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile && !selectedFiles) return

    setIsProcessing(true)
    setError(null)

    const formData = new FormData()

    if (uploadMode === "single" && selectedFile) {
      formData.append("image", selectedFile)
    } else if (uploadMode === "bulk" && selectedFiles) {
      for (let i = 0; i < selectedFiles.length; i++) {
        formData.append("images", selectedFiles[i])
      }
    }

    try {
      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      if (uploadMode === "bulk") {
        setBulkResults(result)
        setResults(null)
      } else {
        setResults(result)
        setBulkResults(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during processing")
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadAnnotatedImage = () => {
    if (!results?.annotated_image) return

    const link = document.createElement("a")
    link.href = `data:image/jpeg;base64,${results.annotated_image}`
    link.download = `runway_detection_${Date.now()}.jpg`
    link.click()
  }

  const downloadAllResults = () => {
    if (!bulkResults?.results) return

    bulkResults.results.forEach((result, index) => {
      if (result.annotated_image) {
        const link = document.createElement("a")
        link.href = `data:image/jpeg;base64,${result.annotated_image}`
        link.download = `bulk_detection_${result.filename || index}_${Date.now()}.jpg`
        link.click()
      }
    })
  }

  const getModeInfo = (mode?: string) => {
    switch (mode) {
      case "production":
        return {
          icon: <CheckCircle className="h-5 w-5 text-green-600" />,
          message: "Real YOLO model detection active",
          color: "bg-green-50 border-green-200 text-green-800",
        }
      case "mock":
      case "bulk":
        return {
          icon: <Info className="h-5 w-5 text-blue-600" />,
          message:
            uploadMode === "bulk"
              ? "Bulk processing completed"
              : "Demo mode - place best.pt in root directory for real detection",
          color: "bg-blue-50 border-blue-200 text-blue-800",
        }
      case "error_fallback":
        return {
          icon: <AlertTriangle className="h-5 w-5 text-orange-600" />,
          message: "YOLO model error - showing demo results",
          color: "bg-orange-50 border-orange-200 text-orange-800",
        }
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="absolute inset-0 bg-white">
        <div className="absolute inset-0 opacity-[0.02]">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `
                linear-gradient(#2563eb 1px, transparent 1px),
                linear-gradient(90deg, #2563eb 1px, transparent 1px)
              `,
              backgroundSize: "60px 60px",
            }}
          />
        </div>

        <div className="absolute top-20 right-20 w-32 h-32 bg-blue-50 rounded-full opacity-30 animate-float"></div>
        <div
          className="absolute bottom-32 left-16 w-24 h-24 bg-cyan-50 rounded-full opacity-40 animate-float"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute top-1/2 right-1/3 w-16 h-16 bg-blue-100 rounded-full opacity-20 animate-float"
          style={{ animationDelay: "4s" }}
        ></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-12">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-6 mb-8">
            <div className="relative group">
              <div className="p-6 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-3xl shadow-lg hover:shadow-xl transition-all duration-300 group-hover:scale-105">
                <Radar className="h-12 w-12 text-white animate-spin" style={{ animationDuration: "8s" }} />
              </div>
            </div>
            <div className="text-left">
              <h1 className="text-6xl font-serif font-bold text-slate-800 mb-2 tracking-tight">SkyWatch AI</h1>
              <p className="text-xl text-blue-600 font-semibold">Precision in Every Pixel</p>
            </div>
          </div>

          <div className="max-w-4xl mx-auto mb-12 bg-slate-50 rounded-2xl p-8 border border-slate-200">
            <h2 className="text-3xl font-serif font-bold text-slate-800 mb-4 leading-tight">
              Transforming Satellite Analysis with AI
            </h2>
            <p className="text-lg text-slate-600 leading-relaxed">
              Explore runway data effortlessly with cutting-edge YOLO deep learning technology. Professional-grade
              analysis for aviation safety and infrastructure monitoring.
            </p>
          </div>
        </div>

        <Card className="mb-16 border border-slate-200 shadow-lg bg-white">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-slate-200">
            <CardTitle className="flex items-center gap-3 text-2xl font-serif font-bold text-slate-800">
              <Upload className="h-7 w-7 text-blue-600" />
              Upload Satellite Images
            </CardTitle>
            <CardDescription className="text-base text-slate-600">
              Select single image or entire folder to detect potential runway structures with AI precision
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8 p-8">
            <div className="flex gap-3 justify-center">
              <Button
                onClick={() => setUploadMode("single")}
                variant={uploadMode === "single" ? "default" : "outline"}
                className={`px-6 py-3 text-base font-semibold rounded-xl transition-all duration-300 ${
                  uploadMode === "single"
                    ? "bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg"
                    : "border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-blue-300"
                }`}
              >
                <Upload className="h-5 w-5 mr-2" />
                Single Image
              </Button>
              <Button
                onClick={() => setUploadMode("bulk")}
                variant={uploadMode === "bulk" ? "default" : "outline"}
                className={`px-6 py-3 text-base font-semibold rounded-xl transition-all duration-300 ${
                  uploadMode === "bulk"
                    ? "bg-cyan-500 hover:bg-cyan-600 text-white shadow-md hover:shadow-lg"
                    : "border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-cyan-300"
                }`}
              >
                <FolderOpen className="h-5 w-5 mr-2" />
                Bulk Upload
              </Button>
            </div>

            <div className="flex flex-col lg:flex-row items-center gap-6">
              <div className="flex-1 w-full">
                {uploadMode === "single" ? (
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="w-full file:mr-4 file:py-4 file:px-6 file:rounded-xl file:border-0 file:text-base file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 file:transition-all file:duration-300 file:shadow-md hover:file:shadow-lg file:cursor-pointer text-slate-600 text-base"
                  />
                ) : (
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    webkitdirectory=""
                    onChange={handleFolderSelect}
                    className="w-full file:mr-4 file:py-4 file:px-6 file:rounded-xl file:border-0 file:text-base file:font-semibold file:bg-cyan-500 file:text-white hover:file:bg-cyan-600 file:transition-all file:duration-300 file:shadow-md hover:file:shadow-lg file:cursor-pointer text-slate-600 text-base"
                  />
                )}
              </div>
              <Button
                onClick={handleUpload}
                disabled={(!selectedFile && !selectedFiles) || isProcessing}
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-700 hover:to-cyan-600 text-white shadow-lg transition-all duration-300 transform hover:scale-105 px-8 py-4 text-base font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                {isProcessing ? (
                  <>
                    <Activity className="h-5 w-5 mr-3 animate-spin" />
                    {uploadMode === "bulk" ? "Processing..." : "Analyzing..."}
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5 mr-3" />
                    {uploadMode === "bulk" ? "Process All Images" : "Detect Runways"}
                    <ArrowRight className="h-5 w-5 ml-2" />
                  </>
                )}
              </Button>
            </div>

            {(selectedFile || selectedFiles) && (
              <div className="space-y-6">
                <div className="text-base text-slate-700 bg-slate-50 p-6 rounded-xl border border-slate-200">
                  {uploadMode === "single" && selectedFile ? (
                    <>
                      <strong className="text-blue-600">Selected File:</strong> {selectedFile.name} (
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </>
                  ) : selectedFiles ? (
                    <>
                      <div className="flex items-center gap-3 mb-3">
                        <Files className="h-5 w-5 text-cyan-500" />
                        <strong className="text-cyan-600">Selected Folder:</strong> {selectedFiles.length} images
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-sm">
                        {Array.from(selectedFiles)
                          .slice(0, 12)
                          .map((file, index) => (
                            <div key={index} className="text-slate-600 truncate">
                              {file.name}
                            </div>
                          ))}
                        {selectedFiles.length > 12 && (
                          <div className="text-cyan-600 font-semibold">+{selectedFiles.length - 12} more...</div>
                        )}
                      </div>
                    </>
                  ) : null}
                </div>

                {previewUrl && uploadMode === "single" && (
                  <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                    <h4 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                      <Eye className="h-5 w-5 text-blue-600" />
                      Image Preview
                    </h4>
                    <div className="w-full max-w-4xl mx-auto border-2 border-slate-200 rounded-xl overflow-hidden shadow-md">
                      <img
                        src={previewUrl || "/placeholder.svg"}
                        alt="Selected satellite image preview"
                        className="w-full h-auto max-h-[500px] object-contain bg-white"
                        style={{ aspectRatio: "auto" }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {error && (
              <div className="p-6 bg-red-50 border-2 border-red-200 rounded-xl text-red-800">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-6 w-6" />
                  <div>
                    <strong className="text-lg">Detection Error:</strong>
                    <p className="mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {(results || bulkResults) && (
          <div className="space-y-12">
            {((results?.mode && getModeInfo(results.mode)) || (bulkResults?.mode && getModeInfo(bulkResults.mode))) && (
              <div
                className={`p-6 rounded-xl border-2 ${getModeInfo(results?.mode || bulkResults?.mode)?.color} flex items-center gap-3`}
              >
                {getModeInfo(results?.mode || bulkResults?.mode)?.icon}
                <span className="font-semibold text-lg">
                  {getModeInfo(results?.mode || bulkResults?.mode)?.message}
                </span>
              </div>
            )}

            {bulkResults && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-blue-50 to-blue-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-blue-600 rounded-xl shadow-md">
                          <Files className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {bulkResults.combined_summary.total_images}
                          </p>
                          <p className="text-blue-700 font-semibold text-sm">Images Processed</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-cyan-50 to-cyan-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-cyan-500 rounded-xl shadow-md">
                          <Target className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {bulkResults.combined_summary.total_detections}
                          </p>
                          <p className="text-cyan-700 font-semibold text-sm">Total Runways</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-indigo-50 to-indigo-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-indigo-600 rounded-xl shadow-md">
                          <Activity className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {bulkResults.combined_summary.average_confidence.toFixed(1)}%
                          </p>
                          <p className="text-indigo-700 font-semibold text-sm">Avg Confidence</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-purple-50 to-purple-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-purple-600 rounded-xl shadow-md">
                          <Zap className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {bulkResults.combined_summary.max_confidence.toFixed(1)}%
                          </p>
                          <p className="text-purple-700 font-semibold text-sm">Peak Confidence</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-green-50 to-green-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-green-600 rounded-xl shadow-md">
                          <CheckCircle className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {bulkResults.combined_summary.images_with_detections}
                          </p>
                          <p className="text-green-700 font-semibold text-sm">With Detections</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card className="border border-slate-200 shadow-lg bg-white">
                  <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-slate-200">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-3 text-2xl font-serif font-bold text-slate-800">
                        <Files className="h-7 w-7 text-blue-600" />
                        Bulk Processing Results
                      </CardTitle>
                      <Button
                        onClick={downloadAllResults}
                        variant="outline"
                        size="lg"
                        className="border-blue-300 text-blue-700 hover:bg-blue-50 hover:border-blue-400 transition-all duration-300 font-semibold px-6 py-3 rounded-xl hover:scale-105 bg-transparent"
                      >
                        <Download className="h-5 w-5 mr-2" />
                        Download All
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="p-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {bulkResults.results.map((result, index) => (
                        <div
                          key={index}
                          className="bg-slate-50 p-5 rounded-xl border border-slate-200 hover:shadow-md transition-all duration-300 hover:scale-[1.02]"
                        >
                          <div className="mb-4">
                            <h4 className="text-lg font-semibold text-slate-800 truncate">
                              {result.filename || `Image ${index + 1}`}
                            </h4>
                            <p className="text-slate-600">
                              {result.summary?.total_detections || 0} detections •{" "}
                              {result.summary?.average_confidence?.toFixed(1) || 0}% avg confidence
                            </p>
                          </div>
                          {result.annotated_image && (
                            <img
                              src={`data:image/jpeg;base64,${result.annotated_image}`}
                              alt={`Detection result for ${result.filename}`}
                              className="w-full h-40 object-cover rounded-lg border border-slate-200"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            {/* Single Image Results */}
            {results && (
              <>
                <Card className="border border-slate-200 shadow-lg bg-white">
                  <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-slate-200">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-3 text-2xl font-serif font-bold text-slate-800">
                        <MapPin className="h-7 w-7 text-blue-600" />
                        Detection Results
                      </CardTitle>
                      <Button
                        onClick={downloadAnnotatedImage}
                        variant="outline"
                        size="lg"
                        className="border-blue-300 text-blue-700 hover:bg-blue-50 hover:border-blue-400 transition-all duration-300 font-semibold px-6 py-3 rounded-xl hover:scale-105 bg-transparent"
                      >
                        <Download className="h-5 w-5 mr-2" />
                        Download Results
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="p-8">
                    <div className="bg-slate-50 rounded-xl overflow-hidden shadow-md border border-slate-200">
                      <img
                        src={`data:image/jpeg;base64,${results.annotated_image}`}
                        alt="AI-annotated detection results"
                        className="w-full h-auto max-h-[600px] object-contain mx-auto"
                      />
                    </div>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-blue-50 to-blue-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-blue-600 rounded-xl shadow-md">
                          <Target className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">{results.summary.total_detections}</p>
                          <p className="text-blue-700 font-semibold text-sm">Runways Detected</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-cyan-50 to-cyan-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-cyan-500 rounded-xl shadow-md">
                          <Activity className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {results.summary.average_confidence.toFixed(1)}%
                          </p>
                          <p className="text-cyan-700 font-semibold text-sm">Average Confidence</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-indigo-50 to-indigo-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-indigo-600 rounded-xl shadow-md">
                          <Zap className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {results.summary.max_confidence.toFixed(1)}%
                          </p>
                          <p className="text-indigo-700 font-semibold text-sm">Peak Confidence</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-purple-50 to-purple-100">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-purple-600 rounded-xl shadow-md">
                          <Satellite className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="text-3xl font-bold text-slate-800 mb-1">
                            {results.summary.image_resolution.width}×{results.summary.image_resolution.height}
                          </p>
                          <p className="text-purple-700 font-semibold text-sm">Resolution</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card className="border border-slate-200 shadow-lg bg-white">
                  <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-slate-200">
                    <CardTitle className="text-2xl font-serif font-bold text-slate-800">Detailed Analysis</CardTitle>
                    <CardDescription className="text-base text-slate-600">
                      Individual runway detections with precise coordinates and AI confidence metrics
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-8">
                    <div className="space-y-8">
                      {results.detections.map((detection, index) => (
                        <div
                          key={index}
                          className="p-8 border-2 rounded-xl border-slate-200 bg-slate-50 shadow-md hover:shadow-lg transition-all duration-300 hover:scale-[1.01]"
                        >
                          <div className="flex items-center justify-between mb-6">
                            <h3 className="font-serif font-bold text-2xl text-slate-800">{detection.label}</h3>
                            <Badge
                              variant="outline"
                              className={`text-base px-6 py-2 font-semibold border-2 rounded-xl ${
                                detection.confidence > 90
                                  ? "bg-green-50 text-green-700 border-green-300"
                                  : detection.confidence > 80
                                    ? "bg-yellow-50 text-yellow-700 border-yellow-300"
                                    : "bg-gray-50 text-gray-700 border-gray-300"
                              }`}
                            >
                              {detection.confidence.toFixed(1)}% Confidence
                            </Badge>
                          </div>

                          <div className="mb-6">
                            <Progress
                              value={detection.confidence}
                              className="h-4 bg-slate-200 rounded-full overflow-hidden"
                            />
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                            <div className="bg-white p-6 rounded-xl border border-slate-200">
                              <span className="font-semibold text-blue-600 text-lg">Coordinates:</span>
                              <p className="text-xl mt-2 text-slate-800">
                                ({detection.position.x}, {detection.position.y})
                              </p>
                            </div>
                            <div className="bg-white p-6 rounded-xl border border-slate-200">
                              <span className="font-semibold text-blue-600 text-lg">Dimensions:</span>
                              <p className="text-xl mt-2 text-slate-800">
                                {detection.dimensions.width} × {detection.dimensions.height} px
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
