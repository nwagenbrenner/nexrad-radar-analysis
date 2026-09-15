#!/usr/bin/env python3

"""
Animate lowest-elevation NEXRAD Level-II reflectivity and radial velocity.

Input:
    Local KGJX NEXRAD Level-II files in:
        kgjx_level2_20260627/

Output:
    kgjx_20260627_0900-1600_MDT.mp4

Important:
    NEXRAD Level-II files can contain separate sweeps at the same nominal
    elevation angle for different moments. Therefore, reflectivity and
    velocity sweeps are selected independently.

For the KGJX data examined here, the lowest positive elevation is about
0.483 degrees, with:
    reflectivity -> sweep 2
    velocity     -> sweep 3

Dependencies:
    pyart
    numpy
    matplotlib
    cartopy
    contextily
    ffmpeg

Tested conceptually against Py-ART 2.x APIs.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx

import pyart


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("kgjx_level2_20260627")

OUTPUT_FILE = Path(
    "kgjx_20260627_0900-1600_MDT.mp4"
)

# Radar fields
REFLECTIVITY_FIELD = "reflectivity"
VELOCITY_FIELD = "velocity"

# Lowest positive elevation to use.
# We will identify this automatically from each radar volume.
MIN_ELEVATION = 0.1

# Plot ranges
REFLECTIVITY_VMIN = -10.0
REFLECTIVITY_VMAX = 70.0

VELOCITY_VMIN = -30.0
VELOCITY_VMAX = 30.0

# Radar display extent.
#
# If None, the extent is determined from the radar coverage in the first
# file. For a fixed scientific animation, it is preferable to use a fixed
# geographic extent.
#
# KGJX is approximately at:
#     39.1 N, -108.5 W
#
# The values below give a fairly large regional view.
MAP_MIN_LON = -111.0
MAP_MAX_LON = -106.0
MAP_MIN_LAT = 37.0
MAP_MAX_LAT = 41.0

# Animation
FPS = 8

# Set this to True if you want every file.
# Set to an integer such as 2 to use every second file while testing.
FRAME_STRIDE = 1

# Contextily zoom adjustment.
# Increase to 1 for more detailed tiles; decrease to -1 for faster plotting.
BASEMAP_ZOOM_ADJUST = 0

# Whether to show the radar location.
SHOW_RADAR_LOCATION = True


# ============================================================================
# Utility functions
# ============================================================================

def parse_timestamp_from_filename(filename):
    """
    Extract UTC timestamp from a NEXRAD Level-II filename.

    Example:
        KGJX20260627_215851_V06
        -> 2026-06-27 21:58:51 UTC
    """

    name = Path(filename).name

    # Expected:
    # KGJXYYYYMMDD_HHMMSS_V06
    timestamp_string = name[4:19]

    dt = datetime.strptime(
        timestamp_string,
        "%Y%m%d_%H%M%S",
    )

    return dt.replace(tzinfo=timezone.utc)


def find_lowest_sweeps(radar):
    """
    Find the lowest positive-elevation sweep containing each field.

    NEXRAD Level-II volumes may have multiple sweeps with the same nominal
    elevation angle. Different moments can be stored in different sweeps.

    Returns
    -------
    reflectivity_sweep : int
    velocity_sweep : int
    elevation : float
        Nominal elevation angle in degrees.
    """

    fixed_angles = np.asarray(
        radar.fixed_angle["data"],
        dtype=float,
    )

    # Find the lowest positive elevation.
    positive_indices = np.where(
        fixed_angles > MIN_ELEVATION
    )[0]

    if len(positive_indices) == 0:
        raise RuntimeError(
            "No positive-elevation sweeps found."
        )

    lowest_angle = fixed_angles[positive_indices[0]]

    # Find all sweeps corresponding to that nominal angle.
    candidates = np.where(
        np.isclose(
            fixed_angles,
            lowest_angle,
            atol=0.01,
        )
    )[0]

    reflectivity_sweep = None
    velocity_sweep = None

    for sweep in candidates:

        # Reflectivity
        if REFLECTIVITY_FIELD in radar.fields:
            refl = radar.get_field(
                sweep,
                REFLECTIVITY_FIELD,
            )

            if np.ma.count(refl) > 0:
                if reflectivity_sweep is None:
                    reflectivity_sweep = int(sweep)

        # Velocity
        if VELOCITY_FIELD in radar.fields:
            vel = radar.get_field(
                sweep,
                VELOCITY_FIELD,
            )

            if np.ma.count(vel) > 0:
                if velocity_sweep is None:
                    velocity_sweep = int(sweep)

    if reflectivity_sweep is None:
        raise RuntimeError(
            f"No valid reflectivity sweep found at "
            f"{lowest_angle:.3f} degrees."
        )

    if velocity_sweep is None:
        raise RuntimeError(
            f"No valid velocity sweep found at "
            f"{lowest_angle:.3f} degrees."
        )

    return (
        reflectivity_sweep,
        velocity_sweep,
        float(lowest_angle),
    )


def print_sweep_diagnostics(radar):
    """
    Print field availability for every sweep.

    This is useful for understanding the NEXRAD Level-II moment structure.
    """

    fixed_angles = np.asarray(
        radar.fixed_angle["data"],
        dtype=float,
    )

    print()
    print("Sweep diagnostics")
    print("-" * 80)
    print(
        f"{'Sweep':>7} "
        f"{'Angle':>9} "
        f"{'Rays':>7} "
        f"{'Reflectivity valid':>22} "
        f"{'Velocity valid':>18}"
    )
    print("-" * 80)

    for sweep in range(radar.nsweeps):

        start = radar.sweep_start_ray_index["data"][sweep]
        end = radar.sweep_end_ray_index["data"][sweep]

        n_rays = int(end - start + 1)

        refl_count = 0
        vel_count = 0

        if REFLECTIVITY_FIELD in radar.fields:
            refl = radar.get_field(
                sweep,
                REFLECTIVITY_FIELD,
            )
            refl_count = int(np.ma.count(refl))

        if VELOCITY_FIELD in radar.fields:
            vel = radar.get_field(
                sweep,
                VELOCITY_FIELD,
            )
            vel_count = int(np.ma.count(vel))

        print(
            f"{sweep:7d} "
            f"{fixed_angles[sweep]:9.3f} "
            f"{n_rays:7d} "
            f"{refl_count:22d} "
            f"{vel_count:18d}"
        )

    print("-" * 80)


def get_radar_location(radar):
    """
    Return radar longitude and latitude.
    """

    lon = float(
        np.asarray(radar.longitude["data"]).flat[0]
    )

    lat = float(
        np.asarray(radar.latitude["data"]).flat[0]
    )

    return lon, lat


# ============================================================================
# Find input files
# ============================================================================

files = sorted(
    DATA_DIR.glob("KGJX*_V06")
)

if not files:
    raise RuntimeError(
        f"No KGJX Level-II files found in {DATA_DIR}"
    )

files = files[::FRAME_STRIDE]

print()
print("=" * 80)
print("KGJX NEXRAD Level-II animation")
print("=" * 80)
print(f"Data directory: {DATA_DIR}")
print(f"Number of files: {len(files)}")
print(f"Output: {OUTPUT_FILE}")
print()

# ============================================================================
# Inspect first file
# ============================================================================

print("Reading first file:")
print(f"  {files[0]}")

radar0 = pyart.io.read_nexrad_archive(
    str(files[0])
)

print()
print(f"Py-ART version: {pyart.__version__}")
print(f"Number of sweeps: {radar0.nsweeps}")

refl_sweep0, vel_sweep0, elevation0 = find_lowest_sweeps(
    radar0
)

print(
    f"Lowest positive elevation: "
    f"{elevation0:.3f} degrees"
)

print(
    f"Reflectivity sweep: {refl_sweep0}"
)

print(
    f"Velocity sweep:     {vel_sweep0}"
)

print()

# Print diagnostics for the first file.
print_sweep_diagnostics(radar0)

radar_lon, radar_lat = get_radar_location(radar0)

print()
print(
    f"Radar location: "
    f"{radar_lat:.4f} N, {radar_lon:.4f} W"
)

# ============================================================================
# Set up figure and map
# ============================================================================

print()
print("Creating figure...")

projection = ccrs.PlateCarree()

fig = plt.figure(
    figsize=(15, 8),
    constrained_layout=True,
)

ax_refl = fig.add_subplot(
    1,
    2,
    1,
    projection=projection,
)

ax_vel = fig.add_subplot(
    1,
    2,
    2,
    projection=projection,
)

# Set geographic extent before adding basemaps.
for ax in (ax_refl, ax_vel):

    ax.set_extent(
        [
            MAP_MIN_LON,
            MAP_MAX_LON,
            MAP_MIN_LAT,
            MAP_MAX_LAT,
        ],
        crs=ccrs.PlateCarree(),
    )

# ============================================================================
# Add basemap
# ============================================================================

print("Adding topographic basemap...")

# Contextily can warp web tiles into the coordinate system of the axes.
#
# WorldTopoMap provides topography plus roads/transportation information,
# which is useful here because we want both terrain context and roads.
#
# The basemap is added ONCE and remains static throughout the animation.
for ax in (ax_refl, ax_vel):

    try:

        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldTopoMap,
            crs="EPSG:4326",
            zoom_adjust=BASEMAP_ZOOM_ADJUST,
            attribution_size=6,
            reset_extent=False,
            alpha=0.75,
            zorder=0,
        )

    except Exception as exc:

        print()
        print(
            "WARNING: Could not download the Contextily "
            "topographic basemap."
        )
        print(f"Reason: {exc}")
        print(
            "The radar plots will still be generated."
        )

# ============================================================================
# Add geographic features
# ============================================================================

for ax in (ax_refl, ax_vel):

    # State boundaries.
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.7,
        edgecolor="black",
        facecolor="none",
        alpha=0.7,
        zorder=3,
    )

    # Coastlines aren't especially useful here, but borders can help with
    # geographic orientation.
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.7,
        edgecolor="black",
        facecolor="none",
        alpha=0.7,
        zorder=3,
    )

    # Radar location.
    if SHOW_RADAR_LOCATION:
        ax.plot(
            radar_lon,
            radar_lat,
            marker="^",
            markersize=8,
            markeredgecolor="black",
            markerfacecolor="white",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

# ============================================================================
# Color maps and colorbars
# ============================================================================

# Py-ART provides radar-specific colormaps.
#
# Fall back to standard matplotlib colormaps if a particular Py-ART
# colormap is not available.
try:
    refl_cmap = pyart.graph.cm.NWSRef
except AttributeError:
    try:
        refl_cmap = pyart.graph.cm.HomeyerRainbow
    except AttributeError:
        refl_cmap = plt.get_cmap("turbo")

try:
    vel_cmap = pyart.graph.cm.NWSVel
except AttributeError:
    vel_cmap = plt.get_cmap("RdBu_r")

refl_norm = Normalize(
    vmin=REFLECTIVITY_VMIN,
    vmax=REFLECTIVITY_VMAX,
)

vel_norm = Normalize(
    vmin=VELOCITY_VMIN,
    vmax=VELOCITY_VMAX,
)

refl_sm = ScalarMappable(
    norm=refl_norm,
    cmap=refl_cmap,
)

vel_sm = ScalarMappable(
    norm=vel_norm,
    cmap=vel_cmap,
)

refl_sm.set_array([])
vel_sm.set_array([])

cbar_refl = fig.colorbar(
    refl_sm,
    ax=ax_refl,
    orientation="vertical",
    pad=0.02,
    fraction=0.046,
)

cbar_refl.set_label(
    "Reflectivity (dBZ)"
)

cbar_vel = fig.colorbar(
    vel_sm,
    ax=ax_vel,
    orientation="vertical",
    pad=0.02,
    fraction=0.046,
)

cbar_vel.set_label(
    "Radial velocity (m s$^{-1}$)"
)

# ============================================================================
# Initial radar plot
# ============================================================================

print()
print("Creating initial radar plots...")

display_refl = pyart.graph.RadarMapDisplay(
    radar0
)

display_vel = pyart.graph.RadarMapDisplay(
    radar0
)

# Py-ART handles the radar polar geometry here rather than us constructing
# a regular longitude/latitude pcolormesh ourselves.
display_refl.plot_ppi_map(
    REFLECTIVITY_FIELD,
    sweep=refl_sweep0,
    ax=ax_refl,
    fig=fig,
    vmin=REFLECTIVITY_VMIN,
    vmax=REFLECTIVITY_VMAX,
    cmap=refl_cmap,
    colorbar_flag=False,
    title_flag=False,
    embellish=False,
    add_grid_lines=False,
    min_lon=MAP_MIN_LON,
    max_lon=MAP_MAX_LON,
    min_lat=MAP_MIN_LAT,
    max_lat=MAP_MAX_LAT,
    edgecolors="face",
    zorder=2,
)

display_vel.plot_ppi_map(
    VELOCITY_FIELD,
    sweep=vel_sweep0,
    ax=ax_vel,
    fig=fig,
    vmin=VELOCITY_VMIN,
    vmax=VELOCITY_VMAX,
    cmap=vel_cmap,
    colorbar_flag=False,
    title_flag=False,
    embellish=False,
    add_grid_lines=False,
    min_lon=MAP_MIN_LON,
    max_lon=MAP_MAX_LON,
    min_lat=MAP_MIN_LAT,
    max_lat=MAP_MAX_LAT,
    edgecolors="face",
    zorder=2,
)

# Keep references to the current radar artists.
current_refl_artists = list(display_refl.plots)
current_vel_artists = list(display_vel.plots)

# ============================================================================
# Gridlines
# ============================================================================

for ax in (ax_refl, ax_vel):

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        alpha=0.5,
        linestyle="--",
        zorder=4,
    )

    gl.top_labels = False
    gl.right_labels = False

# ============================================================================
# Animation update function
# ============================================================================

def update(frame_number):
    """
    Plot one radar volume.
    """

    global current_refl_artists
    global current_vel_artists

    filename = files[frame_number]

    print(
        f"[{frame_number + 1:3d}/{len(files):3d}] "
        f"Reading {filename.name}"
    )

    # ------------------------------------------------------------------------
    # Read radar volume
    # ------------------------------------------------------------------------

    radar = pyart.io.read_nexrad_archive(
        str(filename)
    )

    # ------------------------------------------------------------------------
    # Find field-specific sweeps
    # ------------------------------------------------------------------------

    (
        refl_sweep,
        vel_sweep,
        elevation,
    ) = find_lowest_sweeps(radar)

    # ------------------------------------------------------------------------
    # Remove radar artists from previous frame
    # ------------------------------------------------------------------------

    for artist in current_refl_artists:
        try:
            artist.remove()
        except (ValueError, AttributeError):
            pass

    for artist in current_vel_artists:
        try:
            artist.remove()
        except (ValueError, AttributeError):
            pass

    current_refl_artists = []
    current_vel_artists = []

    # ------------------------------------------------------------------------
    # Create Py-ART display objects
    # ------------------------------------------------------------------------

    display_refl = pyart.graph.RadarMapDisplay(
        radar
    )

    display_vel = pyart.graph.RadarMapDisplay(
        radar
    )

    # ------------------------------------------------------------------------
    # Plot reflectivity
    # ------------------------------------------------------------------------

    display_refl.plot_ppi_map(
        REFLECTIVITY_FIELD,
        sweep=refl_sweep,
        ax=ax_refl,
        fig=fig,
        vmin=REFLECTIVITY_VMIN,
        vmax=REFLECTIVITY_VMAX,
        cmap=refl_cmap,
        colorbar_flag=False,
        title_flag=False,
        embellish=False,
        add_grid_lines=False,
        min_lon=MAP_MIN_LON,
        max_lon=MAP_MAX_LON,
        min_lat=MAP_MIN_LAT,
        max_lat=MAP_MAX_LAT,
        edgecolors="face",
        zorder=2,
    )

    # ------------------------------------------------------------------------
    # Plot velocity
    # ------------------------------------------------------------------------

    display_vel.plot_ppi_map(
        VELOCITY_FIELD,
        sweep=vel_sweep,
        ax=ax_vel,
        fig=fig,
        vmin=VELOCITY_VMIN,
        vmax=VELOCITY_VMAX,
        cmap=vel_cmap,
        colorbar_flag=False,
        title_flag=False,
        embellish=False,
        add_grid_lines=False,
        min_lon=MAP_MIN_LON,
        max_lon=MAP_MAX_LON,
        min_lat=MAP_MIN_LAT,
        max_lat=MAP_MAX_LAT,
        edgecolors="face",
        zorder=2,
    )

    current_refl_artists = list(
        display_refl.plots
    )

    current_vel_artists = list(
        display_vel.plots
    )

    # ------------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------------

    timestamp = parse_timestamp_from_filename(
        filename
    )

    timestamp_mdt = timestamp.astimezone(
        timezone(
            # MDT = UTC-6
            # Using a fixed offset is appropriate for these June 2026 files.
            # No DST transition occurs during the requested period.
            __import__("datetime").timedelta(hours=-6)
        )
    )

    # ------------------------------------------------------------------------
    # Titles
    # ------------------------------------------------------------------------

    ax_refl.set_title(
        "KGJX Reflectivity",
        fontsize=13,
        fontweight="bold",
    )

    ax_vel.set_title(
        "KGJX Radial Velocity",
        fontsize=13,
        fontweight="bold",
    )

    fig.suptitle(
        (
            f"{timestamp_mdt.strftime('%Y-%m-%d %H:%M:%S MDT')}   "
            f"Lowest elevation: {elevation:.3f}°"
        ),
        fontsize=15,
        fontweight="bold",
    )

    # Keep the map extent fixed.
    ax_refl.set_extent(
        [
            MAP_MIN_LON,
            MAP_MAX_LON,
            MAP_MIN_LAT,
            MAP_MAX_LAT,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax_vel.set_extent(
        [
            MAP_MIN_LON,
            MAP_MAX_LON,
            MAP_MIN_LAT,
            MAP_MAX_LAT,
        ],
        crs=ccrs.PlateCarree(),
    )

    return (
        current_refl_artists
        + current_vel_artists
    )


# ============================================================================
# Create animation
# ============================================================================

print()
print("=" * 80)
print("Creating animation")
print("=" * 80)
print(f"Frames: {len(files)}")
print(f"FPS:    {FPS}")
print(f"Output: {OUTPUT_FILE}")
print()

animation = FuncAnimation(
    fig,
    update,
    frames=len(files),
    interval=1000 / FPS,
    blit=False,
    repeat=False,
)

# ============================================================================
# Write MP4
# ============================================================================

print()
print("Writing MP4...")
print(
    "This may take several minutes because each frame "
    "is being rendered with Py-ART and Cartopy."
)
print()

writer = FFMpegWriter(
    fps=FPS,
    metadata={
        "title": "KGJX NEXRAD Level-II Radar Animation",
        "artist": "Py-ART",
        "comment": (
            "Lowest positive-elevation reflectivity and radial velocity "
            "for June 27 2026"
        ),
    },
    bitrate=5000,
)

animation.save(
    OUTPUT_FILE,
    writer=writer,
    dpi=120,
)

plt.close(fig)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Created: {OUTPUT_FILE}")
print()
