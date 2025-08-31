import os
import json
import math
import csv
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import box as shapely_box, mapping as shapely_mapping
from shapely.ops import unary_union
from pyproj import Transformer, CRS
from ultralytics import YOLO
from scipy.ndimage import sobel
from pystac_client import Client
import planetary_computer
from rasterio.warp import transform_bounds as warp_transform_bounds

# ============ CONFIG ============

# Your AOI GeoTIFF file
AOI_TIF = r"aoi\Sentinel_AllBands_Inference_2020_03.tif" 

# The DEM file path is now automatically generated from the AOI filename
aoi_filename = os.path.basename(AOI_TIF)
dem_filename = aoi_filename.replace("Sentinel_AllBands_Inference", "DEM")
DEM_TIF = os.path.join(r"dem", dem_filename)

YOLO_WEIGHTS =r"best.pt" # <-- change to your YOLO model
OUTPUT_DIR = r"outputs"

# tiling (pixels)
TILE_SIZE = 1024 # tile width/height in pixels
OVERLAP = 256# overlap in pixels (helps cross-tile runways)
# YOLO
CONF_THRESH = 0.30
IOU_MERGE_THRESH = 0.3 # merge boxes across tiles if IoU >= this
# DEM filter (DISABLED FOR TESTING)
SLOPE_DEG_MAX = 2.0
FLAT_MIN_PCT = 10.0

# visualization
LABEL_NAME = "Airstrip"
BOX_COLOR = (255, 60, 60)
TEXT_BG = (255, 60, 60)
TEXT_FG = (255, 255, 255)

# ============ DATA STRUCTS ============

@dataclass
class Detection:
    # AOI pixel coordinates
    xmin_px: float
    ymin_px: float
    xmax_px: float
    ymax_px: float
    conf: float
    cls_id: int

    # AOI map coordinates in AOI CRS
    xmin_map: float
    ymin_map: float
    xmax_map: float
    ymax_map: float

    # WGS84 lon/lat
    xmin_lon: float
    ymin_lat: float
    xmax_lon: float
    ymax_lat: float

# ============ UTILS ============

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ["tiles/tif", "tiles/png", "tiles/skipped_dem", "debug"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max stretch to 0..255 safely."""
    arr = np.nan_to_num(arr.astype(np.float32))
    mn, mx = np.min(arr), np.max(arr)
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

def compute_iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def merge_boxes_nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    """Soft NMS-ish: sort by conf, then merge (union) if IoU high and class same."""
    if not dets: return []
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    merged: List[Detection] = []
    used = [False]*len(dets)

    for i, d in enumerate(dets):
        if used[i]:
            continue
        
        group_indices = [i]
        used[i] = True
        
        # Find all other boxes that overlap with the current box
        for j in range(i + 1, len(dets)):
            if used[j] or dets[j].cls_id != d.cls_id:
                continue
            
            iou = compute_iou(
                (d.xmin_px, d.ymin_px, d.xmax_px, d.ymax_px),
                (dets[j].xmin_px, dets[j].ymin_px, dets[j].xmax_px, dets[j].ymax_px)
            )
            
            if iou >= iou_thr:
                group_indices.append(j)
                used[j] = True

        # Now, group all overlapping boxes into one
        group = [dets[idx] for idx in group_indices]
        
        # Merge all boxes in the group geometrically
        polys = [shapely_box(g.xmin_px, g.ymin_px, g.xmax_px, g.ymax_px) for g in group]
        union_poly = unary_union(polys).envelope  # Get the smallest rectangle containing all of them
        minx, miny, maxx, maxy = union_poly.bounds

        # Use the highest confidence score from the group
        best = max(group, key=lambda g: g.conf)

        merged.append(Detection(
            xmin_px=minx, ymin_px=miny, xmax_px=maxx, ymax_px=maxy,
            conf=best.conf, cls_id=best.cls_id,
            xmin_map=0, ymin_map=0, xmax_map=0, ymax_map=0,  # placeholder
            xmin_lon=0, ymin_lat=0, xmax_lon=0, ymax_lat=0
        ))
    return merged


def update_map_coords(dets: List[Detection], aoi_transform: Affine, aoi_crs: CRS) -> List[Detection]:
    """Fill map and WGS84 coords for merged detections."""
    to_wgs84 = Transformer.from_crs(aoi_crs, "EPSG:4326", always_xy=True)

    for d in dets:
        # Pixel (col=x, row=y) → map coords: Affine * (x, y)
        xmin_map_x, ymin_map_y = aoi_transform * (d.xmin_px, d.ymin_px)
        xmax_map_x, ymax_map_y = aoi_transform * (d.xmax_px, d.ymax_px)
        d.xmin_map, d.ymin_map, d.xmax_map, d.ymax_map = xmin_map_x, ymin_map_y, xmax_map_x, ymax_map_y

        # AOI CRS → WGS84
        (d.xmin_lon, d.ymin_lat) = to_wgs84.transform(xmin_map_x, ymin_map_y)
        (d.xmax_lon, d.ymax_lat) = to_wgs84.transform(xmax_map_x, ymax_map_y)
    return dets

# ============ DEM FILTER ============

def is_terrain_flat_enough(dem_data: np.ndarray, dem_nodata: Any,
                           slope_threshold_deg: float = SLOPE_DEG_MAX,
                           min_flat_pct: float = FLAT_MIN_PCT) -> bool:
    """
    Returns True if enough pixels are 'flat' (slope < slope_threshold_deg).
    Takes a numpy array of DEM data directly.
    """
    try:
        dem = dem_data.astype(np.float32)
        if dem_nodata is not None:
            dem[dem == dem_nodata] = np.nan
    except Exception as e:
        print(f"Warning: Could not process DEM data: {e}. Skipping DEM filter.")
        return True

    dx = sobel(dem, axis=1, mode="nearest")
    dy = sobel(dem, axis=0, mode="nearest")
    slope = np.hypot(dx, dy)
    slope_deg = np.degrees(np.arctan(slope))

    valid = np.isfinite(slope_deg)
    if not np.any(valid):
        return True

    flat_pixels = (slope_deg[valid] < slope_threshold_deg)
    flat_pct = float(flat_pixels.mean() * 100.0)
    print(f"[DEM] Flat area: {flat_pct:.2f}% (need >= {min_flat_pct}%)")
    return flat_pct >= min_flat_pct

# ============ DEM DOWNLOADER ============

def download_dem_for_aoi(aoi_path: str, output_dem_path: str):
    """
    Downloads a DEM file for the bounding box of a given AOI GeoTIFF.
    """
    # 1. Get the bounding box from your AOI file
    try:
        with rasterio.open(aoi_path) as src:
            bounds = src.bounds
            aoi_crs = src.crs
            width, height = src.width, src.height
    except Exception as e:
        raise RuntimeError(f"Error reading AOI file at {aoi_path}: {e}")

    # Transform bounds to WGS84 (EPSG:4326) for the API search
    bounds_wgs84 = warp_transform_bounds(aoi_crs, "EPSG:4326", *bounds)
    print(f"Searching for DEM for bounds: {bounds_wgs84}")

    # 2. Connect to the Microsoft Planetary Computer STAC catalog
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # 3. Search for a DEM item within the bounding box
    search = stac.search(
        collections=["cop-dem-glo-30"],  # Using Copernicus GLO-30 DEM
        bbox=bounds_wgs84,
        max_items=1
    )

    items = list(search.get_items())
    if not items:
        raise ValueError("No DEM item found for the specified bounding box.")

    dem_item = items[0]
    print(f"Found DEM: {dem_item.id}")

    # 4. Download and process the DEM data
    signed_url = planetary_computer.sign(dem_item.assets["data"].href)

    with rasterio.open(signed_url) as dem_src:
        dem_data = dem_src.read(1, window=dem_src.window(*bounds_wgs84), 
                                out_shape=(height, width),
                                resampling=rasterio.enums.Resampling.bilinear)
        
        dem_data[dem_data == dem_src.nodata] = np.nan

        # 5. Save the DEM to a new GeoTIFF file
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dem_data.dtype,
            "crs": aoi_crs,
            "transform": rasterio.transform.from_bounds(*bounds, width=width, height=height)
        }

        with rasterio.open(output_dem_path, "w", **profile) as dst:
            dst.write(dem_data, 1)

    print(f"✅ Successfully downloaded and saved DEM to: {output_dem_path}")

# ============ TIF TO PNG CONVERTER ============

def convert_tif_to_png(tif_path, png_path, src):
    """Converts a TIF file to a PNG for YOLO prediction using the src object."""
    try:
        # Use the specific band stacking from your working code
        b2 = src.read(1, boundless=True, fill_value=0)
        b3 = src.read(2, boundless=True, fill_value=0)
        b4 = src.read(3, boundless=True, fill_value=0)

        # Stack as RGB
        rgb = np.stack([b4, b3, b2], axis=-1)
        rgb = np.nan_to_num(rgb)
        
        # Stretch to 0-255 safely
        rgb = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.uint8)
        img = Image.fromarray(rgb, mode='RGB')
        img.save(png_path)
    except Exception as e:
        print(f"Error converting TIF to PNG: {e}")
        return False
    return True

# ============ STEP 1 & 2: TILE + PNG ============

def tile_aoi_to_geo_and_png(aoi_path: str, out_root: str,
                           tile_size: int, overlap: int) -> List[Dict]:
    """
    Returns a list of dicts, one per tile:
    {
      'tif_path', 'png_path', 'window' (x,y,w,h), 'transform', 'crs'
    }
    """
    tiles_info = []
    tif_dir = os.path.join(out_root, "tiles", "tif")
    png_dir = os.path.join(out_root, "tiles", "png")
    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    with rasterio.open(aoi_path) as src:
        W, H = src.width, src.height
        step = tile_size - overlap
        count = 0

        for top in range(0, H, step):
            if top + tile_size > H: top = max(0, H - tile_size)
            for left in range(0, W, step):
                if left + tile_size > W: left = max(0, W - tile_size)

                window = Window(left, top, tile_size, tile_size)
                transform = src.window_transform(window)
                tile_data = src.read(window=window) # shape: (bands, h, w)

                tif_path = os.path.join(tif_dir, f"tile_{count:06d}.tif")
                with rasterio.open(
                    tif_path, "w",
                    driver="GTiff",
                    height=tile_data.shape[1],
                    width=tile_data.shape[2],
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    dst.write(tile_data)
                
                png_path = os.path.join(png_dir, f"tile_{count:06d}.png")
                # Fix: Pass the `src` object to the helper function.
                with rasterio.open(tif_path) as tile_src:
                    convert_tif_to_png(tif_path, png_path, tile_src)

                tiles_info.append({
                    "tif_path": tif_path,
                    "png_path": png_path,
                    "window": (left, top, tile_size, tile_size), # AOI pixel coords
                    "transform": transform,
                    "crs": src.crs.to_string()
                })

                count += 1

            if top + tile_size >= H:
                break

    return tiles_info

# ============ STEP 3 & 4: DEM FILTER + YOLO ============

def run_yolo_on_tiles(tiles_info: List[Dict], yolo_weights: str,
                      conf_thr: float, dem_path: str) -> List[Detection]:
    model = YOLO(yolo_weights)
    detections: List[Detection] = []
    
    # Open the DEM file ONCE before the loop for efficiency
    dem_src = None
    dem_nodata = None
    if dem_path:
        try:
            dem_src = rasterio.open(dem_path)
            dem_nodata = dem_src.nodata
        except Exception as e:
            print(f"Error opening DEM file at {dem_path}: {e}")
            print("All tiles will be processed without DEM filtering.")
            dem_src = None

    for t in tiles_info:
        tif_path = t["tif_path"]
        png_path = t["png_path"]
        left, top, tile_size, _ = t["window"]
        transform: Affine = t["transform"]
        crs = t["crs"]

        # DEM filter on the geotiff tile (skip if not flat enough)
        # This section is now commented out for testing purposes.
        # if dem_src:
        #     try:
        #         dem_data = dem_src.read(1, window=Window(left, top, tile_size, tile_size))
        #         if not is_terrain_flat_enough(dem_data, dem_nodata):
        #             os.replace(
        #                 png_path,
        #                 os.path.join(OUTPUT_DIR, "tiles", "skipped_dem", os.path.basename(png_path))
        #             )
        #             continue
        #     except Exception as e:
        #         print(f"Error processing DEM for tile '{os.path.basename(tif_path)}': {e}. Skipping DEM check for this tile.")
        
        # Now, run YOLO on the PNG image
        results = model(png_path, conf=conf_thr)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy() # (N,4)
            confs = r.boxes.conf.cpu().numpy() # (N,)
            clses = r.boxes.cls.cpu().numpy().astype(int) # (N,)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clses):
                # Convert tile-relative pixels to AOI-pixel coords:
                ax1 = float(left + x1)
                ay1 = float(top + y1)
                ax2 = float(left + x2)
                ay2 = float(top + y2)

                # AOI-pixel → map coords (AOI CRS)
                xmin_map_x, ymin_map_y = transform * (ax1, ay1)
                xmax_map_x, ymax_map_y = transform * (ax2, ay2)

                # AOI CRS → WGS84
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                xmin_lon, ymin_lat = transformer.transform(xmin_map_x, ymin_map_y)
                xmax_lon, ymax_lat = transformer.transform(xmax_map_x, ymax_map_y)

                detections.append(Detection(
                    xmin_px=ax1, ymin_px=ay1, xmax_px=ax2, ymax_px=ay2,
                    conf=float(conf), cls_id=int(cls_id),
                    xmin_map=xmin_map_x, ymin_map=ymin_map_y, xmax_map=xmax_map_x, ymax_map=ymax_map_y,
                    xmin_lon=xmin_lon, ymin_lat=ymin_lat, xmax_lon=xmax_lon, ymax_lat=ymax_lat
                ))

    if dem_src:
        dem_src.close()
    return detections

# ============ STEP 6: MERGE ACROSS TILES ============

def merge_across_tiles(detections: List[Detection],
                       iou_thr: float,
                       aoi_transform: Affine,
                       aoi_crs: CRS) -> List[Detection]:
    merged = merge_boxes_nms(detections, iou_thr)
    merged = update_map_coords(merged, aoi_transform, aoi_crs)
    return merged

# ============ STEP 7: DRAW ON AOI + SAVE TIF & PNG ============

def build_aoi_rgb_preview(aoi_path: str) -> Tuple[Image.Image, Affine, CRS]:
    """Return (PIL RGB image, aoi_transform, aoi_crs)."""
    with rasterio.open(aoi_path) as src:
        aoi_transform = src.transform
        aoi_crs = src.crs
        # Use the specific band combination from the working code
        bands = src.read([3, 2, 1]) # Red, Green, Blue
        
        # Normalize and convert to 8-bit for visualization
        img_array = np.moveaxis(bands, 0, -1)
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            img_array = (255 * (img_array - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    pil_img = Image.fromarray(img_array, mode="RGB")
    return pil_img, aoi_transform, aoi_crs

def draw_detections_on_image(pil_img: Image.Image, dets: List[Detection],
                             label: str = LABEL_NAME) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d.xmin_px, d.ymin_px, d.xmax_px, d.ymax_px
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        tag = f"{label} {i} {d.conf*100:.1f}%"
        tw, th = draw.textbbox((0,0), tag, font=font)[2:]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=TEXT_BG)
        draw.text((x1 + 4, y1 - th - 4), tag, fill=TEXT_FG, font=font)
    return img

def save_rgb_geotiff_like(aoi_path: str, pil_img: Image.Image, out_tif_path: str):
    """Write the drawn RGB image as a georeferenced 3-band GeoTIFF using AOI's transform/CRS."""
    rgb = np.array(pil_img) # (H,W,3) uint8
    H, W, _ = rgb.shape
    with rasterio.open(aoi_path) as src:
        transform = src.transform
        crs = src.crs

    with rasterio.open(
        out_tif_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=3, dtype=rgb.dtype,
        transform=transform, crs=crs
    ) as dst:
        dst.write(rgb[:,:,0], 1)
        dst.write(rgb[:,:,1], 2)
        dst.write(rgb[:,:,2], 3)

# ============ STEP 8: GEOJSON + CSV ============

def export_detections_geojson_csv(dets: List[Detection], out_geojson: str, out_csv: str):
    features = []
    for d in dets:
        geom = shapely_box(d.xmin_lon, d.ymin_lat, d.xmax_lon, d.ymax_lat)
        features.append({
            "type": "Feature",
            "geometry": shapely_mapping(geom),
            "properties": {
                "label": LABEL_NAME,
                "confidence": round(d.conf, 4)
            }
        })
    fc = {"type": "FeatureCollection", "features": features}
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","confidence",
                    "xmin_lon","ymin_lat","xmax_lon","ymax_lat"])
        for d in dets:
            w.writerow([LABEL_NAME, f"{d.conf:.4f}", d.xmin_lon, d.ymin_lat, d.xmax_lon, d.ymax_lat])

# ============ MAIN PIPELINE ============

def main():
    ensure_dirs()
    
    # Create the DEM directory if it doesn't exist
    dem_dir = os.path.dirname(DEM_TIF)
    os.makedirs(dem_dir, exist_ok=True)
    
    # Step 0: Download the DEM file for the AOI
    print("Step 0: Downloading DEM for AOI...")
    try:
        download_dem_for_aoi(AOI_TIF, DEM_TIF)
    except (RuntimeError, ValueError) as e:
        print(f"DEM download failed: {e}. The DEM filter will be disabled.")
        # Proceed with a dummy DEM path that will be ignored by the pipeline
        dem_path_for_pipeline = None
    else:
        dem_path_for_pipeline = DEM_TIF

    # Step 1 & 2
    print("Step 1/2: Tiling AOI and generating PNG tiles…")
    tiles_info = tile_aoi_to_geo_and_png(AOI_TIF, OUTPUT_DIR, TILE_SIZE, OVERLAP)
    print(f"  → {len(tiles_info)} tiles prepared")

    # Step 3 & 4
    print("Step 3/4: DEM filter + YOLO inference on tiles…")
    raw_dets = run_yolo_on_tiles(tiles_info, YOLO_WEIGHTS, CONF_THRESH, dem_path_for_pipeline)
    print(f"  → {len(raw_dets)} raw detections")

    # Step 5 already done inside run_yolo_on_tiles (pixel/map/WGS84 computed)

    # Step 6: Merge across tiles
    print("Step 6: Merging detections across tile boundaries…")
    with rasterio.open(AOI_TIF) as src:
        aoi_transform = src.transform
        aoi_crs = src.crs
    merged = merge_across_tiles(raw_dets, IOU_MERGE_THRESH, aoi_transform, aoi_crs)
    print(f"  → {len(merged)} merged detections")

    # Step 7: Draw on AOI and save GeoTIFF + PNG
    print("Step 7: Drawing final boxes on AOI…")
    aoi_rgb, _, _ = build_aoi_rgb_preview(AOI_TIF)
    drawn = draw_detections_on_image(aoi_rgb, merged, LABEL_NAME)
    png_out = os.path.join(OUTPUT_DIR, "aoi_annotated.png")
    drawn.save(png_out, "PNG")
    tif_out = os.path.join(OUTPUT_DIR, "aoi_annotated_rgb.tif")
    save_rgb_geotiff_like(AOI_TIF, drawn, tif_out)
    print(f"  → Wrote {png_out} and {tif_out}")

    # Step 8: GeoJSON + CSV
    print("Step 8: Exporting GeoJSON and CSV…")
    export_detections_geojson_csv(
        merged,
        os.path.join(OUTPUT_DIR, "detections.geojson"),
        os.path.join(OUTPUT_DIR, "detections.csv")
    )
    print("Done ✅")

if __name__ == "__main__":
    main()
