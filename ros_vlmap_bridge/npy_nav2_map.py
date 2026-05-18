# this script convers a loaded npy occupancy grid as a pgm and a yaml file for use with nav2. The pgm file is a grayscale image where 0 is occupied, 1 is free, 

npy_path = '/tmp/vlmap_recording/test7/map/obstacles.npy'
#!/usr/bin/env python3

import numpy as np
from PIL import Image
import yaml
import argparse
from pathlib import Path


def save_occupancy_map(
    npy_path,
    output_name="from_obs_grid_map",
    resolution=0.02,
    origin=(0.0, 0.0, 0.0),
    occupied_thresh=0.65,
    free_thresh=0.196,
):
    """
    Convert a NumPy occupancy grid to ROS map files (.pgm + .yaml)

    Input occupancy grid:
        0 -> occupied
        1 -> free

    Output PGM values:
        occupied -> 0 (black)
        free     -> 254 (white)

    Parameters
    ----------
    npy_path : str
        Path to .npy occupancy grid
    output_name : str
        Output map name (without extension)
    resolution : float
        Map resolution in meters/pixel
    origin : tuple
        Map origin (x, y, yaw)
    """

    # Load occupancy grid
    grid = np.load(npy_path)

    if grid.ndim != 2:
        raise ValueError("Occupancy grid must be a 2D array")

    # Convert to uint8 image
    # ROS map convention:
    #   0   = occupied (black)
    #   254 = free (white)

    pgm = np.zeros_like(grid, dtype=np.uint8)

    # Free space
    pgm[grid == 1] = 254

    # Occupied space
    pgm[grid == 0] = 0

    # Optional: unknown cells if present
    # Example: grid == -1
    pgm[grid == -1] = 205

    # Flip vertically because image coordinates differ from map coordinates
    # pgm = np.flipud(pgm)

    # Save PGM image
    pgm_path = f"{output_name}.pgm"
    Image.fromarray(pgm).save(pgm_path)

    # Create YAML metadata
    yaml_data = {
        "image": Path(pgm_path).name,
        "mode": "trinary",
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "negate": 0,
        "occupied_thresh": float(occupied_thresh),
        "free_thresh": float(free_thresh),
    }

    yaml_path = f"{output_name}.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"Saved:")
    print(f"  {pgm_path}")
    print(f"  {yaml_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("npy_file", help="Input .npy occupancy grid")
    # parser.add_argument("--output", default="map", help="Output map name")
    # parser.add_argument(
    #     "--resolution",
    #     type=float,
    #     default=0.05,
    #     help="Map resolution in meters/pixel",
    # )

    # args = parser.parse_args()

    save_occupancy_map(
        npy_path,
        output_name="from_obs_grid_map",
        resolution=0.02,
    )