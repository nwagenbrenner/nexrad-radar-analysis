#!/usr/bin/env python3

"""
Download NEXRAD Level-II KGJX data for June 27, 2026,
0900-1600 MDT.

Data source:
    s3://unidata-nexrad-level2/

The script is safe to rerun. Files that have already been downloaded
are skipped.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import s3fs


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RADAR = "KGJX"

# MDT = UTC-6
MDT = timezone(timedelta(hours=-6))

START_MDT = datetime(2026, 6, 27, 9, 0, 0, tzinfo=MDT)
END_MDT = datetime(2026, 6, 27, 16, 0, 0, tzinfo=MDT)

START_UTC = START_MDT.astimezone(timezone.utc)
END_UTC = END_MDT.astimezone(timezone.utc)

DATA_DIR = Path("kgjx_level2_20260627")

BUCKET = "unidata-nexrad-level2"


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def make_s3_filesystem():
    """Create an anonymous S3 filesystem."""
    return s3fs.S3FileSystem(anon=True)


def find_radar_files(fs):
    """
    Find KGJX Level-II files on AWS within the requested time range.
    """
    date_path = (
        f"{BUCKET}/"
        f"{START_UTC:%Y/%m/%d}/"
        f"{RADAR}/"
    )

    pattern = f"{date_path}{RADAR}{START_UTC:%Y%m%d}_*_V06"

    print(f"Searching:")
    print(f"  s3://{pattern}")
    print()

    files = fs.glob(pattern)

    selected = []

    for filename in sorted(files):
        basename = Path(filename).name

        # Expected form:
        # KGJX20260627_HHMMSS_V06
        try:
            timestamp_text = basename.split("_")[0].replace(
                RADAR, "", 1
            )
            timestamp_text = basename.split("_")[0]

            timestamp = datetime.strptime(
                timestamp_text,
                f"{RADAR}%Y%m%d",
            )

            # Extract HHMMSS from the second component.
            time_text = basename.split("_")[1]

            timestamp = timestamp.replace(
                hour=int(time_text[0:2]),
                minute=int(time_text[2:4]),
                second=int(time_text[4:6]),
                tzinfo=timezone.utc,
            )

        except (ValueError, IndexError):
            print(f"  WARNING: Could not parse {basename}")
            continue

        # Skip MDM metadata files if present.
        if "_MDM" in basename:
            continue

        if START_UTC <= timestamp <= END_UTC:
            selected.append(
                {
                    "s3_path": filename,
                    "filename": basename,
                    "timestamp": timestamp,
                }
            )

    return selected


def download_files(fs, files):
    """Download selected radar files, skipping files already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Local data directory:")
    print(f"  {DATA_DIR.resolve()}")
    print()

    downloaded = 0
    skipped = 0

    for item in files:
        local_path = DATA_DIR / item["filename"]

        if local_path.exists():
            print(f"Already exists: {local_path.name}")
            skipped += 1
            continue

        print(f"Downloading: {item['filename']}")

        fs.get(
            item["s3_path"],
            str(local_path),
        )

        downloaded += 1

    print()
    print("Download summary:")
    print(f"  Files found:      {len(files)}")
    print(f"  Files downloaded: {downloaded}")
    print(f"  Files skipped:    {skipped}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KGJX NEXRAD Level-II Downloader")
    print("=" * 70)
    print()
    print(f"Radar:       {RADAR}")
    print(
        f"Time range:  {START_MDT:%Y-%m-%d %H:%M:%S} "
        f"to {END_MDT:%Y-%m-%d %H:%M:%S} MDT"
    )
    print(
        f"UTC range:   {START_UTC:%Y-%m-%d %H:%M:%S} "
        f"to {END_UTC:%Y-%m-%d %H:%M:%S} UTC"
    )
    print()

    fs = make_s3_filesystem()

    files = find_radar_files(fs)

    if not files:
        raise RuntimeError(
            "No KGJX Level-II files were found for the requested "
            "time range."
        )

    print(f"Found {len(files)} radar files:")
    for item in files:
        print(
            f"  {item['timestamp']:%Y-%m-%d %H:%M:%S UTC}  "
            f"{item['filename']}"
        )

    print()

    download_files(fs, files)


if __name__ == "__main__":
    main()

