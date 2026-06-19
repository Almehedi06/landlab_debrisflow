from pathlib import Path
import runpy


# Simple v1 sequence.
# Each step still uses the paths/parameters defined at the top of that script.

repo_dir = Path(__file__).resolve().parent

scripts = [
    "01_build_erosion_evidence.py",
    "02_build_slope_area_index.py",
    "03_build_coherent_zones.py",
    "04_filter_small_objects.py",
    "05_filter_zones_by_headwater_streams.py",
    "06_filter_zones_by_slope_area_index.py",
]

for script_name in scripts:
    script_path = repo_dir / script_name
    print("\n" + "=" * 80)
    print("running:", script_path)
    print("=" * 80)
    runpy.run_path(script_path, run_name="__main__")

print("\nV1 pipeline complete.")
