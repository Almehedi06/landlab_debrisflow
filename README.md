# debris-landlab

Landlab-focused workspace for postfire terrain, ecohydrology, and landslide probability workflows.

The production code lives in `src/`. Notebooks are kept for exploration, checks, and visualization, but the multi-model landslide probability workflow is now runnable from a YAML config and a script.

## Layout

- `config/base.yaml`: shared config for older workflow experiments
- `config/mmp_landslide.yaml`: modular forcing, snow, ecohydrology, and landslide probability run config
- `config/scenarios/`: scenario-specific YAML overrides
- `notebook/`: exploratory and legacy notebooks, kept intact
- `scripts/run_mmp_landslide.py`: script entrypoint for the modular MMP landslide workflow
- `scripts/run_landlab_batch.py`: parallel batch runner for terrain evolution experiments
- `src/debris_landlab/components/`: project-local Landlab component variants
- `src/debris_landlab/mmp/`: modular MMP workflow code grouped by notebook section
- `src/prism_forcing.py`: PRISM download, clip, align, and ASC export utility

## Environment

Conda:

```bash
conda env create -f environment.yml
conda activate debris-landlab
```

Pip:

```bash
pip install -e ".[dev,viz]"
```

## Run MMP Landslide Probability

The old `MMP_LS.ipynb` workflow is split into modules:

- `terrain.py`: DEM loading, boundary setup, flow accumulation
- `static_inputs.py`: soil, vegetation, burn severity, transmissivity, and cohesion fields
- `daily_forcing.py`: prepared PRISM forcing loading and nodata sanitization
- `snow.py`: rain/snow partitioning, SWE, and melt
- `ecohydrology.py`: daily PET and soil moisture
- `landslides.py`: routed recharge and Landlab landslide probability
- `exports.py`: optional GeoTIFF and ASC outputs

Prepare PRISM forcing once for the AOI, date window, and DEM grid:

```bash
build-prism-forcing \
  --start-date 2025-12-07 \
  --end-date 2025-12-20 \
  --dem-path /mnt/c/Users/amehedi/Downloads/pioneer/output/topographic__elevation.asc \
  --aoi-path /mnt/c/Users/amehedi/Downloads/pioneer/huc_pioneer1.shp \
  --output-dir /mnt/c/Users/amehedi/Downloads/pioneer/output/prism_forcing \
  --dem-crs EPSG:32610
```

The MMP pipeline does not download PRISM data. It reads the prepared
`prism_forcing/forcing_daily_prism.csv` index and the ASC files referenced by that index.
If the index is missing or incomplete, it can also discover the standard
`prism_forcing/asc/{ppt,tmin,tmax}/` files by date.

Run the model from the repo root after all scenario inputs are present:

```bash
python scripts/run_mmp_landslide.py \
  --config config/mmp_landslide.yaml \
  --summary-json experiments/mmp_landslide/summary.json
```

With an installed editable package, the console command is also available:

```bash
run-mmp-landslide --config config/mmp_landslide.yaml
```

Use small YAML overrides for alternate date windows, parameters, or export settings:

```bash
python scripts/run_mmp_landslide.py \
  --config config/mmp_landslide.yaml \
  --override config/scenarios/mmp_cohesion_burnsev_reduction.yaml
```

## Plot A Result In A Notebook

After running the modular pipeline in a notebook:

```python
from debris_landlab.mmp import load_mmp_config, run_pipeline

cfg = load_mmp_config("../config/mmp_landslide.yaml")
result = run_pipeline(cfg)
grid = result.grid
```

Plot landslide probability:

```python
from landlab.plot.imshow import imshow_grid_at_node

imshow_grid_at_node(
    grid,
    "landslide__probability_of_failure",
    plot_name="Landslide Probability of Failure",
    var_name="LS probability",
    var_units="[-]",
    grid_units=("m", "m"),
    cmap="magma_r",
    vmin=0,
    vmax=1,
)
```

## Other Utilities

Resolve config files:

```bash
resolve-workflow-config \
  --base config/base.yaml \
  --override config/scenarios/cohesion_burnsev_reduction.yaml \
  --format yaml
```

Run terrain evolution batch experiments:

```bash
python scripts/run_landlab_batch.py \
  --dem-path /mnt/c/Users/amehedi/Downloads/nsf_rapid/asc/BoltCreek_USGS_1m_DEM_Reference_A.asc \
  --n-runs 4 \
  --max-workers 4 \
  --total-t 10 \
  --dt 1 \
  --out-dir experiments/landlab_batch
```

Export ASC rasters to GeoTIFF and Zarr:

```bash
export-raster-products \
  --output-dir data \
  --overwrite \
  --crs EPSG:32610 \
  --zarr-store experiments/outputs/layers.zarr
```
