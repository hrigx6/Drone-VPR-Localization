import os
import json
import time
import requests
from pathlib import Path
from math import floor, log, tan, pi, cos

load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
ZOOM        = 18          # retrieval database zoom (76m tiles, matches 61m drone FOV)
ZOOM_FINE   = 20          # pulled on-demand for SuperGlue refinement
TILE_SIZE   = 640         # pixels per tile
OUTPUT_DIR  = Path("data/boston/tiles_z18")
META_FILE   = Path("data/boston/tiles_z18/metadata.json")

# JP / Roxbury / Northeastern bounding box
NORTH =  42.3380
SOUTH =  42.3150
EAST  = -71.0550
WEST  = -71.1100


def lat_lon_to_tile(lat, lon, zoom):
    """Convert GPS coordinate to tile X/Y at given zoom."""
    n    = 2 ** zoom
    xtile = int((lon + 180) / 360 * n)
    ytile = int((1 - log(tan(lat * pi / 180) + 1 / cos(lat * pi / 180)) / pi) / 2 * n)
    return xtile, ytile


def tile_to_lat_lon(x, y, zoom):
    """Convert tile X/Y to center GPS coordinate."""
    n    = 2 ** zoom
    lon  = x / n * 360 - 180
    lat  = (x / n * 360 - 180)  # placeholder, compute properly below
    import math
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def download_tile(x, y, zoom, output_path):
    """Download one Mapbox satellite tile."""
    # Mapbox Static Tiles API
    url = (
        f"https://api.mapbox.com/v4/mapbox.satellite"
        f"/{zoom}/{x}/{y}@2x.jpg90"
        f"?access_token={MAPBOX_TOKEN}"
    )
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"  Failed: {response.status_code} for tile {x},{y}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # compute tile range for bounding box
    x_min, y_max = lat_lon_to_tile(SOUTH, WEST, ZOOM)
    x_max, y_min = lat_lon_to_tile(NORTH, EAST, ZOOM)

    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"Zoom level  : {ZOOM}")
    print(f"Tile range  : x={x_min}-{x_max}, y={y_min}-{y_max}")
    print(f"Total tiles : {total}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Est. cost   : ~${total * 0.004:.2f}")
    print()

    metadata = {}
    downloaded = 0
    skipped    = 0
    failed     = 0

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_name = f"tile_{ZOOM}_{x}_{y}.jpg"
            tile_path = OUTPUT_DIR / tile_name

            # skip if already downloaded
            if tile_path.exists():
                skipped += 1
                lat, lon = tile_to_lat_lon(x, y, ZOOM)
                metadata[tile_name] = {"lat": lat, "lon": lon, "x": x, "y": y, "zoom": ZOOM}
                continue

            lat, lon = tile_to_lat_lon(x, y, ZOOM)

            success = download_tile(x, y, ZOOM, tile_path)
            if success:
                metadata[tile_name] = {
                    "lat"  : lat,
                    "lon"  : lon,
                    "x"    : x,
                    "y"    : y,
                    "zoom" : ZOOM,
                }
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  Downloaded {downloaded}/{total - skipped} tiles...")
            else:
                failed += 1

            # rate limit — Mapbox allows 600 req/min
            time.sleep(0.1)

    # save metadata
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone.")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  Failed     : {failed}")
    print(f"  Metadata   : {META_FILE}")
    print(f"  Total tiles: {len(metadata)}")


if __name__ == "__main__":
    main()
