import json
import xml.etree.ElementTree as ET
from pathlib import Path

KML_DIR  = Path("/home/ved/workspace/vpr/data/university1652-first-key/first-key")
OUT_FILE = Path("configs/gps_index.json")
OUT_FILE.parent.mkdir(exist_ok=True)

def parse_kml(kml_path):
    """
    Extract lon, lat from a single KML file.
    Coordinates tag format: longitude,latitude,altitude
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML uses a namespace — must include it in tag search
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords = root.find(".//kml:coordinates", ns)

    if coords is None:
        return None

    parts = coords.text.strip().split(",")
    lon = float(parts[0])
    lat = float(parts[1])
    return lat, lon


if __name__ == "__main__":
    gps_index = {}
    missing   = []

    kml_files = sorted(KML_DIR.glob("*.kml"))
    print(f"Found {len(kml_files)} KML files")

    for kml_path in kml_files:
        building_id = kml_path.stem          # filename without .kml e.g. '0000'
        result = parse_kml(kml_path)

        if result:
            lat, lon = result
            gps_index[building_id] = {"lat": lat, "lon": lon}
        else:
            missing.append(building_id)

    print(f"Parsed   : {len(gps_index)} buildings")
    print(f"Missing  : {len(missing)}")

    with open(OUT_FILE, "w") as f:
        json.dump(gps_index, f, indent=2)

    print(f"Saved → {OUT_FILE}")

    # quick sanity check
    print(f"\nSample entries:")
    for bid in ["0000", "0001", "0002"]:
        if bid in gps_index:
            g = gps_index[bid]
            print(f"  {bid}: lat={g['lat']:.6f}, lon={g['lon']:.6f}")
