import rasterio
from PIL import Image
import numpy as np
import os

def convert_multi_band_tif_to_png(input_file_path, output_file_path):
    """
    Converts a multi-band GeoTIFF image to a single-band PNG by selecting and normalizing one band.
    
    Args:
        input_file_path (str): The path to the input multi-band TIF file.
        output_file_path (str): The path where the converted PNG image will be saved.
    """
    try:
        with rasterio.open(input_file_path) as src:
            # Check if the file has multiple bands
            if src.count > 1:
                print(f"File has {src.count} bands. Converting a single band to PNG...")
                
                # Read the first band (index 1) from the GeoTIFF
                band_data = src.read(1)

                # Normalize the band data to an 8-bit range (0-255) for visualization
                band_data = (band_data - band_data.min()) / (band_data.max() - band_data.min())
                band_data_8bit = (band_data * 255).astype(np.uint8)

                # Create a Pillow image from the numpy array
                img = Image.fromarray(band_data_8bit, 'L')  # 'L' mode is for single-band grayscale

                # Save the image as a PNG
                img.save(output_file_path, "PNG")
                print(f"Successfully converted band 1 of '{input_file_path}' to '{output_file_path}'")

            else:
                print("The file has only one band. Using the standard converter...")
                # If there's only one band, you could fall back to a simpler method
                img = Image.open(input_file_path)
                img.save(output_file_path, "PNG")
                print(f"Successfully converted '{input_file_path}' to '{output_file_path}'")

    except Exception as e:
        print(f"Failed to convert the file: {e}")

# Example usage:
input_file_path = r"Zindi_Amazon_Results (Sentinel_inf)\output\tile_0_1024.tif"
output_file_path = r"C:\Users\spsuh\Desktop\zindi\Zindi_Amazon_Results (Sentinel_inf)\drive-download-20250814T093044Z-1-001\Sentinel_AllBands_Inference_2020_02.png"

convert_multi_band_tif_to_png(input_file_path, output_file_path)