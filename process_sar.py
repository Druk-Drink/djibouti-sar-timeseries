#!/usr/bin/env python3
"""
Sentinel-1 GRD Preprocessing Script using SNAP GPT

Processes downloaded SAR data using the standard workflow:
1. Apply Orbit File
2. Thermal Noise Removal
3. Calibration (Sigma0)
4. Speckle Filtering (Refined Lee 7x7)
5. Terrain Correction (SRTM 1Sec, WGS84)

Outputs GeoTIFF files ready for visualization/analysis.

Requirements:
- SNAP Desktop installed (https://step.esa.int/main/download/snap-download/)
- GPT executable in PATH or set GPT_PATH below

Usage:
    python process_sar.py
"""

import os
import subprocess
import platform
from glob import glob
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# SNAP GPT executable path
# macOS: /Applications/snap/bin/gpt
# Linux: /usr/local/snap/bin/gpt or ~/snap/bin/gpt
# Windows: C:/Program Files/snap/bin/gpt.exe
# GPT path: Set via environment variable or auto-detect
# Set SNAP_GPT_PATH in .env file, or leave as None for auto-detection
GPT_PATH = os.getenv('SNAP_GPT_PATH', None)

# Processing graph XML file
GRAPH_XML = './grd_preprocessing.xml'

# Input/Output directories
INPUT_BASE_PATH = './djibouti_sar_data'
OUTPUT_BASE_PATH = './djibouti_sar_processed'

# Processing options
SKIP_EXISTING = True  # Skip if output already exists
MAX_MEMORY = '8G'  # Max memory for GPT (adjust based on your system)


# =============================================================================
# Functions
# =============================================================================

def find_gpt():
    """Find GPT executable path."""
    if GPT_PATH and os.path.exists(GPT_PATH):
        return GPT_PATH

    # Common installation paths
    system = platform.system()
    candidates = []

    if system == 'Darwin':  # macOS
        candidates = [
            '/Applications/snap/bin/gpt',
            os.path.expanduser('~/Applications/snap/bin/gpt'),
        ]
    elif system == 'Linux':
        candidates = [
            '/usr/local/snap/bin/gpt',
            os.path.expanduser('~/snap/bin/gpt'),
            '/opt/snap/bin/gpt',
        ]
    elif system == 'Windows':
        candidates = [
            'C:/Program Files/snap/bin/gpt.exe',
            'C:/snap/bin/gpt.exe',
        ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Try to find in PATH
    try:
        result = subprocess.run(
            ['which', 'gpt'] if system != 'Windows' else ['where', 'gpt'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    return None


def process_scene(gpt_path, graph_xml, input_file, output_file):
    """
    Process a single SAR scene using GPT.

    Args:
        gpt_path: Path to GPT executable
        graph_xml: Path to processing graph XML
        input_file: Input .zip or .SAFE file
        output_file: Output GeoTIFF path

    Returns:
        True if successful, False otherwise
    """
    # Build GPT command
    cmd = [
        gpt_path,
        graph_xml,
        f'-Pinput={input_file}',
        f'-Poutput={output_file}',
        f'-J-Xmx{MAX_MEMORY}',
        '-x',  # Clear cache after processing
    ]

    print(f"    Command: {' '.join(cmd[:3])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per scene
        )

        if result.returncode == 0:
            return True
        else:
            print(f"    ERROR: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("    ERROR: Processing timeout (>1 hour)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def get_output_filename(input_zip):
    """Generate output filename from input zip."""
    # Extract base name: S1A_IW_GRDH_..._XXXX.zip -> S1A_IW_GRDH_..._XXXX_TC.tif
    basename = Path(input_zip).stem
    return f"{basename}_TC.tif"


# =============================================================================
# Main Script
# =============================================================================

def main():
    print("=" * 70)
    print("Sentinel-1 GRD Preprocessing with SNAP GPT")
    print("=" * 70)

    # Find GPT
    print("\n[1/4] Locating SNAP GPT...")
    gpt_path = find_gpt()

    if not gpt_path:
        print("  ERROR: SNAP GPT not found!")
        print("\n  Please install SNAP Desktop from:")
        print("  https://step.esa.int/main/download/snap-download/")
        print("\n  Or set GPT_PATH in this script manually.")
        return

    print(f"  Found: {gpt_path}")

    # Check graph XML exists
    if not os.path.exists(GRAPH_XML):
        print(f"\n  ERROR: Graph XML not found: {GRAPH_XML}")
        return

    # Find all input files
    print("\n[2/4] Finding SAR scenes to process...")
    input_files = []

    for month_dir in sorted(glob(os.path.join(INPUT_BASE_PATH, '*/'))):
        for direction in ['ascending', 'descending']:
            dir_path = os.path.join(month_dir, direction)
            if os.path.exists(dir_path):
                zips = glob(os.path.join(dir_path, '*.zip'))
                input_files.extend(zips)

    if not input_files:
        print("  No .zip files found in input directory.")
        return

    print(f"  Found {len(input_files)} scenes to process")

    # Create output directory structure
    print("\n[3/4] Setting up output directories...")
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # Process each scene
    print("\n[4/4] Processing scenes...")
    print("-" * 70)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, input_file in enumerate(sorted(input_files)):
        # Determine output path (mirror input structure)
        rel_path = os.path.relpath(os.path.dirname(input_file), INPUT_BASE_PATH)
        output_dir = os.path.join(OUTPUT_BASE_PATH, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = get_output_filename(input_file)
        output_file = os.path.join(output_dir, output_filename)

        print(f"\n[{i+1}/{len(input_files)}] {os.path.basename(input_file)}")

        # Skip if exists
        if SKIP_EXISTING and os.path.exists(output_file):
            print("    Skipped (output exists)")
            skip_count += 1
            continue

        # Process
        print(f"    Output: {output_file}")
        success = process_scene(gpt_path, GRAPH_XML, input_file, output_file)

        if success:
            print("    Done!")
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"  Successful: {success_count}")
    print(f"  Skipped:    {skip_count}")
    print(f"  Failed:     {fail_count}")
    print(f"\n  Output: {OUTPUT_BASE_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
