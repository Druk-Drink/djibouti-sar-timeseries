#!/usr/bin/env python3
"""
SAR Time Series GIF Generator

Creates animated GIF from processed Sentinel-1 SAR images with:
- Title text (top-left)
- Date overlay (top-right)
- Scale bar (bottom-left)
- Optional AOI cropping with EPSG:3857 coordinates

Usage:
    python create_gif.py
"""

import os
import re
from glob import glob
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

# =============================================================================
# Configuration
# =============================================================================

# Input/Output paths (now using processed GeoTIFFs)
BASE_DATA_PATH = './djibouti_sar_processed'
TARGET_MONTH = 1  # Must match download_sar.py setting
OUTPUT_GIF_PATH = './gifs'

# =============================================================================
# AOI Configuration (Area of Interest)
# =============================================================================

# Set to True to crop to AOI, False to use full image
USE_AOI = True

# AOI bounds in EPSG:3857 (Web Mercator) - format: [xmin, xmax, ymin, ymax]
# Example: Djibouti city area
AOI_BOUNDS_3857 = [4827473.4918, 4839230.1208, 1416579.1557, 1424199.4877]

# AOI name for output filename (no spaces)
AOI_NAME = "mayyun"

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
SCALE_BAR_KM = 3  # Length of scale bar in kilometers (adjust based on AOI size)
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

def convert_aoi_to_wgs84(bounds_3857):
    """
    Convert AOI bounds from EPSG:3857 to EPSG:4326 (WGS84).

    Args:
        bounds_3857: [xmin, xmax, ymin, ymax] in EPSG:3857

    Returns:
        [lon_min, lon_max, lat_min, lat_max] in WGS84
    """
    transformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
    xmin, xmax, ymin, ymax = bounds_3857
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)
    return [lon_min, lon_max, lat_min, lat_max]


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
    Format: S1A_IW_GRDH_1SDV_20230115T030405_..._TC.tif
    """
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None


def read_processed_geotiff(tif_path, polarization='VV', aoi_bounds_wgs84=None):
    """
    Read processed SAR image from GeoTIFF.

    Args:
        tif_path: Path to processed GeoTIFF file
        polarization: 'VV' or 'VH' (band 1 or 2)
        aoi_bounds_wgs84: Optional [lon_min, lon_max, lat_min, lat_max] for cropping

    Returns:
        numpy array, acquisition date, pixel resolution in meters
    """
    # Extract date from filename
    date = extract_date_from_filename(os.path.basename(tif_path))

    with rasterio.open(tif_path) as src:
        # Select band (1 = VV, 2 = VH based on SNAP output)
        band_idx = 1 if polarization.upper() == 'VV' else 2

        # Get pixel resolution
        transform = src.transform
        pixel_size = abs(transform[0])  # degrees per pixel

        # Convert to meters (approximate at Djibouti latitude ~12.5°N)
        lat_center = 12.5
        meters_per_degree = 111320 * np.cos(np.radians(lat_center))
        pixel_size_m = pixel_size * meters_per_degree

        if aoi_bounds_wgs84 is not None:
            # Crop to AOI
            lon_min, lon_max, lat_min, lat_max = aoi_bounds_wgs84

            # Create window from bounds
            window = from_bounds(
                left=lon_min, bottom=lat_min,
                right=lon_max, top=lat_max,
                transform=src.transform
            )

            # Read windowed data
            data = src.read(band_idx, window=window)
        else:
            # Read full image
            data = src.read(band_idx)

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


def add_date(draw, img_width, date, missing_year=None):
    """
    Add date text at top-right corner.
    If missing_year is provided, show "No Data for YYYY" below the date.
    """
    font = get_font(FONT_SIZE_MEDIUM)
    font_small = get_font(FONT_SIZE_SMALL)
    date_text = date.strftime('%Y-%m-%d')

    bbox = draw.textbbox((0, 0), date_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate "No Data" text dimensions if needed
    no_data_text = f"No Data for {missing_year}" if missing_year else None
    no_data_width = 0
    no_data_height = 0
    if no_data_text:
        no_data_bbox = draw.textbbox((0, 0), no_data_text, font=font_small)
        no_data_width = no_data_bbox[2] - no_data_bbox[0]
        no_data_height = no_data_bbox[3] - no_data_bbox[1]

    max_text_width = max(text_width, no_data_width)
    total_height = text_height + (no_data_height + 4 if missing_year else 0)

    x = img_width - max_text_width - PADDING * 2
    y = PADDING

    # Draw background
    bg_rect = [
        x - PADDING,
        y - PADDING // 2,
        x + max_text_width + PADDING,
        y + total_height + PADDING // 2
    ]
    draw.rectangle(bg_rect, fill=TEXT_BACKGROUND)

    # Draw date text (right-aligned)
    date_x = x + (max_text_width - text_width)
    draw.text((date_x, y), date_text, font=font, fill=TEXT_COLOR)

    # Draw "No Data" text in orange if applicable
    if no_data_text:
        no_data_x = x + (max_text_width - no_data_width)
        no_data_y = y + text_height + 4
        draw.text((no_data_x, no_data_y), no_data_text, font=font_small, fill=(255, 180, 80))


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


def add_overlays(img, date, title_text, pixel_size_m, scale_factor, missing_year=None):
    """
    Add all overlays: title, date, scale bar, and optional "No Data" indicator.

    Args:
        img: PIL Image
        date: datetime of the actual image
        title_text: Title string
        pixel_size_m: Pixel size in meters
        scale_factor: Image resize scale factor
        missing_year: If set, indicates this image is a substitute for the missing year
    """
    # Convert to RGBA for transparency support
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    draw = ImageDraw.Draw(img)

    # Add title (top-left)
    add_title(draw, img.width, title_text)

    # Add date (top-right), with optional "No Data" indicator
    add_date(draw, img.width, date, missing_year)

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
    print(f"Source: Processed GeoTIFFs ({BASE_DATA_PATH})")
    print("=" * 70)

    # Convert AOI bounds if using AOI
    aoi_bounds_wgs84 = None
    if USE_AOI:
        aoi_bounds_wgs84 = convert_aoi_to_wgs84(AOI_BOUNDS_3857)
        print(f"\nAOI enabled: {AOI_NAME}")
        print(f"  EPSG:3857: x=[{AOI_BOUNDS_3857[0]:.2f}, {AOI_BOUNDS_3857[1]:.2f}], "
              f"y=[{AOI_BOUNDS_3857[2]:.2f}, {AOI_BOUNDS_3857[3]:.2f}]")
        print(f"  WGS84: lon=[{aoi_bounds_wgs84[0]:.6f}, {aoi_bounds_wgs84[1]:.6f}], "
              f"lat=[{aoi_bounds_wgs84[2]:.6f}, {aoi_bounds_wgs84[3]:.6f}]")

    # Process both ascending and descending
    for direction in ['ascending', 'descending']:
        data_path = os.path.join(BASE_DATA_PATH, month_folder, direction)

        print(f"\n[{direction.upper()}]")
        print(f"  Looking for data in: {data_path}")

        if not os.path.exists(data_path):
            print("  Directory not found. Skipping.")
            continue

        # Find all processed GeoTIFF files (*_TC.tif)
        tif_files = sorted(glob(os.path.join(data_path, '*_TC.tif')))

        if not tif_files:
            print("  No processed GeoTIFF files found. Skipping.")
            continue

        print(f"  Found {len(tif_files)} processed SAR images")

        # Generate title for this set
        if USE_AOI:
            title_text = f"{AOI_NAME.replace('_', ' ').title()} - {month_name} ({direction.capitalize()})"
        else:
            title_text = TITLE_TEMPLATE.format(
                month=month_name,
                direction=direction.capitalize()
            )

        # Process each image - store raw data for later reuse
        raw_frames = {}  # year -> (data, date, pixel_size_m, normalized, img, scale_factor)
        for i, tif_path in enumerate(tif_files):
            filename = os.path.basename(tif_path)
            print(f"  Processing [{i+1}/{len(tif_files)}]: {filename[:50]}...")

            try:
                # Read processed SAR data with optional AOI cropping
                data, date, pixel_size_m = read_processed_geotiff(
                    tif_path, POLARIZATION, aoi_bounds_wgs84
                )
                if data is None or data.size == 0:
                    print(f"    Warning: No data returned for {filename}")
                    continue

                # Normalize for visualization
                normalized = normalize_sar(data, PERCENTILE_MIN, PERCENTILE_MAX)

                # Resize
                img, scale_factor = resize_image(normalized, IMAGE_WIDTH)

                # Store for this year
                year = date.year
                raw_frames[year] = (date, pixel_size_m, img.copy(), scale_factor)

            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                continue

        if not raw_frames:
            print("  No valid frames. Skipping.")
            continue

        # Determine full year range
        min_year = min(raw_frames.keys())
        max_year = max(raw_frames.keys())
        all_years = list(range(min_year, max_year + 1))

        print(f"  Year range: {min_year}-{max_year}")
        print(f"  Available years: {sorted(raw_frames.keys())}")
        missing_years = [y for y in all_years if y not in raw_frames]
        if missing_years:
            print(f"  Missing years (will use previous): {missing_years}")

        # Build final frames, filling missing years with previous year's image
        frames = []
        last_available_data = None

        for year in all_years:
            if year in raw_frames:
                # Use actual data for this year
                date, pixel_size_m, img, scale_factor = raw_frames[year]
                img_with_overlays = add_overlays(
                    img.copy(), date, title_text, pixel_size_m, scale_factor, missing_year=None
                )
                frames.append((year, img_with_overlays))
                last_available_data = raw_frames[year]
            else:
                # Missing year - use previous year's image
                if last_available_data:
                    date, pixel_size_m, img, scale_factor = last_available_data
                    img_with_overlays = add_overlays(
                        img.copy(), date, title_text, pixel_size_m, scale_factor, missing_year=year
                    )
                    frames.append((year, img_with_overlays))
                    print(f"    Year {year}: Using {date.year} image (No Data)")

        # Sort by year (already sorted, but ensure)
        frames.sort(key=lambda x: x[0])
        sorted_images = [f[1] for f in frames]

        # Create output directory
        os.makedirs(OUTPUT_GIF_PATH, exist_ok=True)

        # Save GIF with AOI name if applicable
        if USE_AOI:
            output_file = os.path.join(
                OUTPUT_GIF_PATH,
                f'{AOI_NAME}_{month_folder}_{direction}_{POLARIZATION}.gif'
            )
        else:
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
