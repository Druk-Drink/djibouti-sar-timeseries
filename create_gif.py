#!/usr/bin/env python3
"""
SAR Time Series GIF Generator

Creates animated GIF from downloaded Sentinel-1 SAR images with:
- Title text (top-left)
- Date overlay (top-right)
- Scale bar (bottom-left)

Usage:
    python create_gif.py
"""

import os
import re
import zipfile
from glob import glob
from datetime import datetime

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

# =============================================================================
# Configuration
# =============================================================================

# Input/Output paths
BASE_DATA_PATH = './djibouti_sar_data'
TARGET_MONTH = 1  # Must match download_sar.py setting
OUTPUT_GIF_PATH = './gifs'

# Month names mapping
MONTH_NAMES = {
    1: '01_january', 2: '02_february', 3: '03_march', 4: '04_april',
    5: '05_may', 6: '06_june', 7: '07_july', 8: '08_august',
    9: '09_september', 10: '10_october', 11: '11_november', 12: '12_december'
}

# GIF settings
GIF_FPS = 1  # Frames per second (1 = 1 second per image)
IMAGE_WIDTH = 800  # Output image width in pixels

# Font settings
FONT_SIZE_LARGE = 28  # For title
FONT_SIZE_MEDIUM = 24  # For date
FONT_SIZE_SMALL = 18  # For scale bar label
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BACKGROUND = (0, 0, 0, 180)  # Semi-transparent black
PADDING = 10

# Title settings
TITLE_TEMPLATE = "Djibouti SAR - {month} ({direction})"  # {month} and {direction} will be replaced

# Scale bar settings
SCALE_BAR_KM = 20  # Length of scale bar in kilometers
SCALE_BAR_HEIGHT = 8  # Height in pixels
SCALE_BAR_COLOR = (255, 255, 255)  # White
SCALE_BAR_OUTLINE = (0, 0, 0)  # Black outline

# SAR visualization
POLARIZATION = 'VV'  # Options: 'VV', 'VH'
PERCENTILE_MIN = 2  # For contrast stretch
PERCENTILE_MAX = 98


# =============================================================================
# Functions
# =============================================================================

def get_font(size):
    """Load a font with fallbacks."""
    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',  # macOS
        '/System/Library/Fonts/SFNSMono.ttf',  # macOS
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_date_from_filename(filename):
    """
    Extract acquisition date from Sentinel-1 filename.
    Format: S1A_IW_GRDH_1SDV_20230115T030405_...
    """
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None


def find_measurement_tiff(zip_path, polarization='VV'):
    """
    Find the measurement GeoTIFF inside Sentinel-1 SAFE zip file.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if 'measurement' in name and name.endswith('.tiff'):
                if f'-{polarization.lower()}-' in name.lower():
                    return name
    return None


def read_sar_from_zip(zip_path, polarization='VV'):
    """
    Read SAR image data from zipped SAFE format.
    Returns numpy array, acquisition date, and pixel resolution in meters.
    """
    tiff_name = find_measurement_tiff(zip_path, polarization)
    if not tiff_name:
        print(f"  Warning: No {polarization} measurement found in {zip_path}")
        return None, None, None

    # Extract date from filename
    date = extract_date_from_filename(os.path.basename(zip_path))

    # Read GeoTIFF from inside zip
    zip_uri = f'zip://{zip_path}!/{tiff_name}'
    with rasterio.open(zip_uri) as src:
        data = src.read(1)
        # Get pixel resolution (assumes square pixels, in degrees for GRD)
        transform = src.transform
        pixel_size = abs(transform[0])  # degrees per pixel

        # Convert to meters (approximate at Djibouti latitude ~11.5°N)
        # 1 degree latitude ≈ 111,320 m
        # 1 degree longitude ≈ 111,320 * cos(lat) m
        lat_center = 11.5
        meters_per_degree = 111320 * np.cos(np.radians(lat_center))
        pixel_size_m = pixel_size * meters_per_degree

    return data, date, pixel_size_m


def normalize_sar(data, percentile_min=2, percentile_max=98):
    """
    Normalize SAR data to 0-255 using percentile stretch.
    Converts to dB scale first for better visualization.
    """
    # Avoid log of zero
    data = np.where(data > 0, data, np.nan)

    # Convert to dB
    data_db = 10 * np.log10(data)

    # Get percentile values (ignoring NaN)
    vmin = np.nanpercentile(data_db, percentile_min)
    vmax = np.nanpercentile(data_db, percentile_max)

    # Normalize to 0-255
    normalized = (data_db - vmin) / (vmax - vmin) * 255
    normalized = np.clip(normalized, 0, 255)
    normalized = np.nan_to_num(normalized, nan=0)

    return normalized.astype(np.uint8)


def resize_image(img_array, target_width):
    """
    Resize image maintaining aspect ratio.
    Returns resized image and scale factor.
    """
    img = Image.fromarray(img_array)
    scale_factor = target_width / img.width
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized, scale_factor


def add_title(draw, img_width, title_text):
    """
    Add title text at top-left corner.
    """
    font = get_font(FONT_SIZE_LARGE)

    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = PADDING
    y = PADDING

    # Draw background
    bg_rect = [
        x - PADDING,
        y - PADDING // 2,
        x + text_width + PADDING,
        y + text_height + PADDING // 2
    ]
    draw.rectangle(bg_rect, fill=TEXT_BACKGROUND)

    # Draw text
    draw.text((x, y), title_text, font=font, fill=TEXT_COLOR)

    return text_height + PADDING * 2


def add_date(draw, img_width, date):
    """
    Add date text at top-right corner.
    """
    font = get_font(FONT_SIZE_MEDIUM)
    date_text = date.strftime('%Y-%m-%d')

    bbox = draw.textbbox((0, 0), date_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = img_width - text_width - PADDING * 2
    y = PADDING

    # Draw background
    bg_rect = [
        x - PADDING,
        y - PADDING // 2,
        x + text_width + PADDING,
        y + text_height + PADDING // 2
    ]
    draw.rectangle(bg_rect, fill=TEXT_BACKGROUND)

    # Draw text
    draw.text((x, y), date_text, font=font, fill=TEXT_COLOR)


def add_scale_bar(draw, img_width, img_height, pixel_size_m, scale_factor):
    """
    Add scale bar at bottom-left corner.

    Args:
        draw: ImageDraw object
        img_width: Image width in pixels
        img_height: Image height in pixels
        pixel_size_m: Original pixel size in meters
        scale_factor: Resize scale factor
    """
    font = get_font(FONT_SIZE_SMALL)

    # Calculate scale bar length in pixels
    # After resize, each pixel represents: pixel_size_m / scale_factor meters
    meters_per_pixel = pixel_size_m / scale_factor
    scale_bar_m = SCALE_BAR_KM * 1000  # Convert km to meters
    scale_bar_px = int(scale_bar_m / meters_per_pixel)

    # Limit scale bar width to reasonable size
    max_width = img_width // 3
    if scale_bar_px > max_width:
        # Adjust km value
        actual_km = (max_width * meters_per_pixel) / 1000
        scale_bar_px = max_width
        label_text = f"{actual_km:.0f} km"
    else:
        label_text = f"{SCALE_BAR_KM} km"

    # Position (bottom-left)
    x = PADDING * 2
    y = img_height - PADDING * 2 - SCALE_BAR_HEIGHT

    # Get label dimensions
    label_bbox = draw.textbbox((0, 0), label_text, font=font)
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]

    # Background rectangle
    bg_rect = [
        x - PADDING,
        y - label_height - PADDING,
        x + max(scale_bar_px, label_width) + PADDING,
        y + SCALE_BAR_HEIGHT + PADDING
    ]
    draw.rectangle(bg_rect, fill=TEXT_BACKGROUND)

    # Draw scale bar outline
    bar_rect = [x, y, x + scale_bar_px, y + SCALE_BAR_HEIGHT]
    draw.rectangle(bar_rect, fill=SCALE_BAR_COLOR, outline=SCALE_BAR_OUTLINE)

    # Draw alternating pattern (makes it look more professional)
    segment_width = scale_bar_px // 4
    for i in range(4):
        if i % 2 == 1:
            seg_rect = [
                x + i * segment_width,
                y,
                x + (i + 1) * segment_width,
                y + SCALE_BAR_HEIGHT
            ]
            draw.rectangle(seg_rect, fill=SCALE_BAR_OUTLINE)

    # Draw label centered above scale bar
    label_x = x + (scale_bar_px - label_width) // 2
    label_y = y - label_height - 4
    draw.text((label_x, label_y), label_text, font=font, fill=TEXT_COLOR)


def add_overlays(img, date, title_text, pixel_size_m, scale_factor):
    """
    Add all overlays: title, date, and scale bar.
    """
    # Convert to RGBA for transparency support
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    draw = ImageDraw.Draw(img)

    # Add title (top-left)
    add_title(draw, img.width, title_text)

    # Add date (top-right)
    add_date(draw, img.width, date)

    # Add scale bar (bottom-left)
    if pixel_size_m:
        add_scale_bar(draw, img.width, img.height, pixel_size_m, scale_factor)

    return img


def create_gif(frames, output_path, fps=1):
    """
    Create animated GIF from list of PIL images.
    """
    # Convert RGBA to RGB (GIF doesn't support alpha well)
    rgb_frames = []
    for frame in frames:
        if frame.mode == 'RGBA':
            bg = Image.new('RGB', frame.size, (0, 0, 0))
            bg.paste(frame, mask=frame.split()[3])
            rgb_frames.append(np.array(bg))
        else:
            rgb_frames.append(np.array(frame.convert('RGB')))

    # Save GIF
    duration = int(1000 / fps)
    iio.imwrite(
        output_path,
        rgb_frames,
        extension='.gif',
        duration=duration,
        loop=0
    )


# =============================================================================
# Main Script
# =============================================================================

def main():
    month_folder = MONTH_NAMES[TARGET_MONTH]
    month_name = month_folder.split('_')[1].capitalize()

    print("=" * 70)
    print("SAR Time Series GIF Generator")
    print(f"Month: {month_name}")
    print("=" * 70)

    # Process both ascending and descending
    for direction in ['ascending', 'descending']:
        data_path = os.path.join(BASE_DATA_PATH, month_folder, direction)

        print(f"\n[{direction.upper()}]")
        print(f"  Looking for data in: {data_path}")

        if not os.path.exists(data_path):
            print("  Directory not found. Skipping.")
            continue

        # Find all zip files
        zip_files = sorted(glob(os.path.join(data_path, '*.zip')))

        if not zip_files:
            print("  No .zip files found. Skipping.")
            continue

        print(f"  Found {len(zip_files)} SAR images")

        # Generate title for this set
        title_text = TITLE_TEMPLATE.format(
            month=month_name,
            direction=direction.capitalize()
        )

        # Process each image
        frames = []
        for i, zip_path in enumerate(zip_files):
            filename = os.path.basename(zip_path)
            print(f"  Processing [{i+1}/{len(zip_files)}]: {filename[:50]}...")

            # Read SAR data
            data, date, pixel_size_m = read_sar_from_zip(zip_path, POLARIZATION)
            if data is None:
                continue

            # Normalize for visualization
            normalized = normalize_sar(data, PERCENTILE_MIN, PERCENTILE_MAX)

            # Resize
            img, scale_factor = resize_image(normalized, IMAGE_WIDTH)

            # Add all overlays
            img_with_overlays = add_overlays(
                img, date, title_text, pixel_size_m, scale_factor
            )

            frames.append((date, img_with_overlays))

        if not frames:
            print("  No valid frames. Skipping.")
            continue

        # Sort by date
        frames.sort(key=lambda x: x[0])
        sorted_images = [f[1] for f in frames]

        # Create output directory
        os.makedirs(OUTPUT_GIF_PATH, exist_ok=True)

        # Save GIF
        output_file = os.path.join(
            OUTPUT_GIF_PATH,
            f'djibouti_{month_folder}_{direction}_{POLARIZATION}.gif'
        )

        print(f"\n  Creating GIF with {len(sorted_images)} frames...")
        create_gif(sorted_images, output_file, GIF_FPS)
        print(f"  Saved: {output_file}")

    print("\n" + "=" * 70)
    print("GIF generation complete!")
    print(f"Output directory: {OUTPUT_GIF_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
