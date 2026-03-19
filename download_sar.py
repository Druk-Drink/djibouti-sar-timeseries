#!/usr/bin/env python3
"""
Sentinel-1 SAR Data Download Script for Djibouti
For time series GIF analysis (2014-2026)

Features:
- Configurable target month (expandable to all 12 months)
- Fixed orbit tracks for consistent geometry
- Organized folder structure for GIF generation

NASA Earthdata credentials are loaded from environment variables.
"""

import os
from datetime import datetime
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv
import asf_search as asf

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Configuration - EDIT THESE PARAMETERS
# =============================================================================

# Target month for analysis (1=January, 2=February, ..., 12=December)
# Start with one month, later expand to others
TARGET_MONTH = 1  # January

# NASA Earthdata credentials (from environment variables)
EARTHDATA_USERNAME = os.getenv('EARTHDATA_USERNAME')
EARTHDATA_PASSWORD = os.getenv('EARTHDATA_PASSWORD')

# Djibouti bounding box (WKT format)
DJIBOUTI_WKT = """POLYGON((
    41.75 10.95,
    43.45 10.95,
    43.45 12.72,
    41.75 12.72,
    41.75 10.95
))"""

# Date range (Sentinel-1 operational from October 2014)
START_YEAR = 2014
END_YEAR = 2026

# Download settings
BASE_DOWNLOAD_PATH = './djibouti_sar_data'
PARALLEL_DOWNLOADS = 4

# Month names for folder structure
MONTH_NAMES = {
    1: '01_january', 2: '02_february', 3: '03_march', 4: '04_april',
    5: '05_may', 6: '06_june', 7: '07_july', 8: '08_august',
    9: '09_september', 10: '10_october', 11: '11_november', 12: '12_december'
}


# =============================================================================
# Functions
# =============================================================================

def analyze_orbits(results):
    """
    Analyze available orbits and count scenes per orbit/direction.
    """
    orbit_stats = defaultdict(lambda: {'ASCENDING': 0, 'DESCENDING': 0, 'years': set()})

    for scene in results:
        props = scene.properties
        orbit = props.get('pathNumber')
        direction = props.get('flightDirection', 'UNKNOWN')
        date = datetime.fromisoformat(props['startTime'].replace('Z', '+00:00'))

        if orbit and direction in ['ASCENDING', 'DESCENDING']:
            orbit_stats[orbit][direction] += 1
            orbit_stats[orbit]['years'].add(date.year)

    return orbit_stats


def get_best_orbits(orbit_stats):
    """
    Select the best ascending and descending orbits based on coverage.
    """
    best_asc = None
    best_desc = None
    max_asc = 0
    max_desc = 0

    for orbit, stats in orbit_stats.items():
        if stats['ASCENDING'] > max_asc:
            max_asc = stats['ASCENDING']
            best_asc = orbit
        if stats['DESCENDING'] > max_desc:
            max_desc = stats['DESCENDING']
            best_desc = orbit

    return best_asc, best_desc


def filter_by_month_and_orbit(results, target_month, asc_orbit, desc_orbit):
    """
    Filter results to only include scenes from target month and specified orbits.
    Select one image per year per orbit direction (closest to mid-month).
    """
    # Group by year and direction
    grouped = defaultdict(lambda: {'ASCENDING': [], 'DESCENDING': []})

    for scene in results:
        props = scene.properties
        orbit = props.get('pathNumber')
        direction = props.get('flightDirection')
        date = datetime.fromisoformat(props['startTime'].replace('Z', '+00:00'))

        # Filter by month and orbit
        if date.month != target_month:
            continue
        if not ((orbit == asc_orbit and direction == 'ASCENDING') or
                (orbit == desc_orbit and direction == 'DESCENDING')):
            continue

        grouped[date.year][direction].append({
            'scene': scene,
            'date': date,
            'props': props,
            'orbit': orbit
        })

    # Select one image per year per direction (closest to day 15 of month)
    selected = []

    for year in sorted(grouped.keys()):
        for direction in ['ASCENDING', 'DESCENDING']:
            scenes = grouped[year][direction]
            if scenes:
                # Select scene closest to mid-month (day 15)
                target_date = datetime(year, target_month, 15, tzinfo=scenes[0]['date'].tzinfo)
                closest = min(scenes, key=lambda x: abs((x['date'] - target_date).days))
                selected.append({
                    'year': year,
                    'month': target_month,
                    'direction': direction,
                    'orbit': closest['orbit'],
                    'scene': closest['scene'],
                    'date': closest['date'],
                    'props': closest['props']
                })

    return selected


def create_metadata_dataframe(selected_scenes):
    """Create a pandas DataFrame with metadata of selected scenes."""
    data = []
    for item in selected_scenes:
        props = item['props']
        data.append({
            'year': item['year'],
            'month': item['month'],
            'orbit': item['orbit'],
            'direction': item['direction'],
            'date': item['date'].strftime('%Y-%m-%d'),
            'file_id': props['fileID'],
            'polarization': props.get('polarization', 'N/A'),
            'size_gb': round(props['bytes'] / (1024**3), 2),
        })
    return pd.DataFrame(data)


# =============================================================================
# Main Script
# =============================================================================

def main():
    # Validate credentials
    if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
        print("=" * 70)
        print("ERROR: NASA Earthdata credentials not found!")
        print("=" * 70)
        print("\nPlease create a .env file with:")
        print("  EARTHDATA_USERNAME=your_username")
        print("  EARTHDATA_PASSWORD=your_password")
        print("\nRegister at: https://urs.earthdata.nasa.gov/users/new")
        return

    month_name = MONTH_NAMES[TARGET_MONTH].split('_')[1].capitalize()

    print("=" * 70)
    print(f"Sentinel-1 SAR Download - Djibouti Time Series")
    print(f"Target: {month_name} ({START_YEAR}-{END_YEAR})")
    print("=" * 70)

    # Authenticate
    print("\n[1/6] Authenticating with NASA Earthdata...")
    try:
        session = asf.ASFSession().auth_with_creds(
            EARTHDATA_USERNAME,
            EARTHDATA_PASSWORD
        )
        print("  ✓ Authentication successful")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        return

    # Search for all SAR data (to analyze orbit coverage)
    print("\n[2/6] Searching for Sentinel-1 GRD_HD data...")
    results = asf.search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel='GRD_HD',
        start=datetime(START_YEAR, 1, 1),
        end=datetime(END_YEAR, 12, 31),
        intersectsWith=DJIBOUTI_WKT,
    )
    print(f"  Total scenes in database: {len(results)}")

    if len(results) == 0:
        print("  No data found for Djibouti.")
        return

    # Analyze orbits
    print("\n[3/6] Analyzing orbit coverage...")
    orbit_stats = analyze_orbits(results)

    print("\n  Available orbits:")
    print("  " + "-" * 50)
    print(f"  {'Orbit':<8} {'Ascending':<12} {'Descending':<12} {'Years':<15}")
    print("  " + "-" * 50)
    for orbit in sorted(orbit_stats.keys()):
        stats = orbit_stats[orbit]
        year_range = f"{min(stats['years'])}-{max(stats['years'])}"
        print(f"  {orbit:<8} {stats['ASCENDING']:<12} {stats['DESCENDING']:<12} {year_range:<15}")

    # Select best orbits
    best_asc, best_desc = get_best_orbits(orbit_stats)
    print("\n  Recommended orbits (most coverage):")
    print(f"    Ascending:  {best_asc}")
    print(f"    Descending: {best_desc}")

    # User confirmation
    choice = input("\n  Use these orbits? [Y/n] or enter 'asc,desc' (e.g., '101,79'): ").strip()
    if choice.lower() == 'n':
        print("  Cancelled.")
        return
    elif choice and ',' in choice:
        parts = choice.split(',')
        best_asc = int(parts[0].strip())
        best_desc = int(parts[1].strip())
        print(f"  Using: Ascending={best_asc}, Descending={best_desc}")

    # Filter for target month
    print(f"\n[4/6] Filtering for {month_name} images...")
    selected = filter_by_month_and_orbit(results, TARGET_MONTH, best_asc, best_desc)

    if not selected:
        print(f"  No images found for {month_name}.")
        return

    # Create and display metadata
    df = create_metadata_dataframe(selected)

    asc_count = len(df[df['direction'] == 'ASCENDING'])
    desc_count = len(df[df['direction'] == 'DESCENDING'])

    print(f"\n  Selected scenes:")
    print(f"    Ascending:  {asc_count} images (orbit {best_asc})")
    print(f"    Descending: {desc_count} images (orbit {best_desc})")
    print(f"    Total: {len(selected)} images")

    print("\n" + "-" * 70)
    print(df.to_string(index=False))
    print("-" * 70)

    total_gb = df['size_gb'].sum()
    print(f"\n  Total download size: {total_gb:.2f} GB")

    # Setup folder structure
    month_folder = MONTH_NAMES[TARGET_MONTH]
    asc_path = os.path.join(BASE_DOWNLOAD_PATH, month_folder, 'ascending')
    desc_path = os.path.join(BASE_DOWNLOAD_PATH, month_folder, 'descending')

    print(f"\n  Download structure:")
    print(f"    {asc_path}/")
    print(f"    {desc_path}/")

    # Save metadata
    metadata_path = os.path.join(BASE_DOWNLOAD_PATH, month_folder, 'metadata.csv')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    df.to_csv(metadata_path, index=False)
    print(f"\n  Metadata saved: {metadata_path}")

    # Confirm download
    confirm = input("\n[5/6] Proceed with download? [y/N]: ").strip()
    if confirm.lower() != 'y':
        print("  Download cancelled. Metadata saved for reference.")
        return

    # Create directories
    os.makedirs(asc_path, exist_ok=True)
    os.makedirs(desc_path, exist_ok=True)

    # Separate ascending and descending scenes
    asc_scenes = [item['scene'] for item in selected if item['direction'] == 'ASCENDING']
    desc_scenes = [item['scene'] for item in selected if item['direction'] == 'DESCENDING']

    # Download ascending
    print(f"\n[6/6] Downloading...")
    if asc_scenes:
        print(f"\n  Downloading {len(asc_scenes)} ascending scenes...")
        asc_results = asf.ASFSearchResults(asc_scenes)
        asc_results.download(path=asc_path, session=session, processes=PARALLEL_DOWNLOADS)

    # Download descending
    if desc_scenes:
        print(f"\n  Downloading {len(desc_scenes)} descending scenes...")
        desc_results = asf.ASFSearchResults(desc_scenes)
        desc_results.download(path=desc_path, session=session, processes=PARALLEL_DOWNLOADS)

    print("\n" + "=" * 70)
    print("Download complete!")
    print("=" * 70)
    print(f"\nData saved to:")
    print(f"  {asc_path}/ ({asc_count} files)")
    print(f"  {desc_path}/ ({desc_count} files)")
    print(f"\nNext steps:")
    print(f"  1. Process SAR images (calibration, terrain correction)")
    print(f"  2. Generate GIF animation")
    print(f"  3. Change TARGET_MONTH and re-run for other months")
    print("=" * 70)


if __name__ == '__main__':
    main()
