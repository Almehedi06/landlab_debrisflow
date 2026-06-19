# Landslide Probability Technical Document

This note documents the current landslide workflow implemented in `notebook/Multi_Model_Probability_updated1.ipynb` and its helper scripts in `notebook/`.

The goal is to describe the actual notebook behavior: inputs, derived fields, transformations, equations, model flow, outputs, and current caveats.

## 1. Scope

The notebook combines:

- DEM-based terrain processing in Landlab
- Imported soil, vegetation, and cohesion rasters
- Daily forcing for a storm window
- Rain/snow partition and simple SWE bookkeeping
- Daily ecohydrologic simulation using custom `Radiation`, `PotentialEvapotranspiration`, and `SoilMoisture` components
- Recharge aggregation and optional recharge routing
- Monte Carlo landslide probability using Landlab `LandslideProbability`

Main notebook studied here:

- `notebook/Multi_Model_Probability_updated1.ipynb`

Main helper scripts used by the notebook:

- `notebook/soil_moisture_dynamics1.py`
- `notebook/radiation_field_OFFICIAL.py`
- `notebook/potential_evapotranspiration_field_OFFICIAL.py`
- `notebook/recharge_routing.py`

## 2. Runtime Assumptions

The notebook assumes the working directory is explicitly changed to:

```text
/mnt/c/Users/amehedi/Downloads/ml_debris/pioneer/output/cut1
```

This directory is treated as the main data root for the ASC rasters and some forcing products.

The notebook also appends the repo notebook directory to `sys.path`:

```text
/home/abdullah/landlab_debrisflow/notebook
```

That path is required so the notebook can import:

- `soil_moisture_dynamics1`
- `recharge_routing`

## 3. Required Input Data

### 3.1 Static raster inputs in the working directory

These files are read from the `cut1` directory:

| File | Field added to grid | Role |
|---|---|---|
| `topographic__elevation.asc` | `topographic__elevation` | DEM and base grid |
| `soil__thickness.asc` | `soil__thickness` | Soil thickness |
| `soil__density.asc` | `soil__density` | Soil bulk density |
| `soil__internal_friction_angle.asc` | `soil__internal_friction_angle` | Friction angle for infinite-slope stability |
| `porosity.asc` | `porosity` | Porosity |
| `field__capacity.asc` | `field__capacity` | Volumetric field capacity |
| `wilting__point.asc` | `wilting__point` | Volumetric wilting point |
| `soil__saturated_hydraulic_conductivity.asc` | `soil__saturated_hydraulic_conductivity` | Saturated hydraulic conductivity |
| `vegetation__plant_functional_type.asc` | `vegetation__plant_functional_type` | Vegetation/PFT control |
| `soil__maximum_total_cohesion.asc` | `soil__maximum_total_cohesion` | Max total cohesion |
| `soil__mode_total_cohesion.asc` | `soil__mode_total_cohesion` | Mode total cohesion |
| `soil__minimum_total_cohesion.asc` | `soil__minimum_total_cohesion` | Min total cohesion |

### 3.2 Disturbance layer

This file is read from outside the working directory:

```text
/mnt/c/Users/amehedi/Downloads/burn__severity.asc
```

It is added as:

- `burn__severity`

### 3.3 Forcing files

The notebook currently contains two forcing pathways.

#### A. CSV-based daily forcing

Read from current working directory:

- `precip_2026-01.csv`
- `temp_2026-01.csv`

The code expects columns:

- `datetime`
- `precip_mm`
- `tmin_c`
- `tmax_c`

#### B. Gridded PRISM forcing

Read from:

```text
<asc_dir>/prism_daily_asc/ppt
<asc_dir>/prism_daily_asc/tmin
<asc_dir>/prism_daily_asc/tmax
```

For dates:

- `2025-12-07` to `2025-12-20`

These are later used in the ecohydrology run.

### 3.4 External calibration / comparison data

The notebook also reads SMAP files from Downloads:

- `/mnt/c/Users/amehedi/Downloads/SMAPL4_SM_post_fire_mudflow_daily.csv`
- `/mnt/c/Users/amehedi/Downloads/SMAPL4_SM_post_fire_mudflow2_daily_pacific.csv`
- `/mnt/c/Users/amehedi/Downloads/SMAPL4_SM_post_fire_mudflow2_pacific.csv`

These are used for initialization and later diagnostic comparison.

### 3.5 Export template

GeoTIFF export uses:

```text
/mnt/c/Users/amehedi/Downloads/ml_debris/pioneer/output/cut1/topographic__elevation.tif
```

as the template raster profile.

## 4. Grid and Terrain Setup

### 4.1 DEM load and boundary handling

The DEM is loaded into a Landlab raster grid using `esri_ascii.load`.

Nodes with DEM nodata value `-999999` are closed:

```math
z_i = \text{closed if } z_i = -999999
```

Low-elevation open nodes with elevation `<= 337.0` are reassigned to fixed-value boundary condition. Then D8 flow routing with depression handling is run.

### 4.2 Derived terrain fields

The notebook derives and/or stores:

- `drainage_area`
- `log_drainage_area`
- `topographic__slope`
- `Aspect`
- `Hillshade`
- `topographic__specific_contributing_area`

Specific contributing area is computed as:

```math
a = \frac{A_d}{\Delta x}
```

where:

- `a` = specific contributing area
- `A_d` = drainage area
- `\Delta x` = grid cell width

## 5. Soil, Vegetation, and Disturbance Fields

### 5.1 Soil hydraulic fields

Imported node fields:

- `porosity`
- `field__capacity`
- `wilting__point`
- `soil__saturated_hydraulic_conductivity`

These are volumetric hydraulic properties, except conductivity.

### 5.2 Transmissivity

The notebook derives transmissivity as:

```math
T = K_{sat} \times 2.5 \times h_s
```

where:

- `T` = `soil__transmissivity`
- `K_sat` = saturated hydraulic conductivity
- `2.5` = anisotropy multiplier used in the notebook
- `h_s` = soil thickness

A positive floor is enforced:

```math
T = \max(T, 0.01)
```

### 5.3 Vegetation and LAI

The PFT raster is mapped from node to cell because the ecohydrology component operates on cells.

The notebook creates a simple LAI lookup:

- grass: `1.5`
- shrub: `2.0`
- tree: `4.0`
- bare: `1.0`

This becomes:

- `vegetation__live_leaf_area_index`

Vegetation cover fraction is later set as:

```math
f_{cover} = \frac{LAI}{4}
```

### 5.4 Cohesion fields

Three cohesion rasters are imported:

- `soil__minimum_total_cohesion`
- `soil__mode_total_cohesion`
- `soil__maximum_total_cohesion`

These are the bounds used by Landlab `LandslideProbability` for triangular sampling.

### 5.5 Burn severity

The notebook imports `burn__severity` and uses it for masking and for tentative post-fire cohesion logic.

However, in the current active notebook state, burn severity is **not actually reducing cohesion in the final landslide run**. The `post_fire_cohesion` function returns a multiplier of `1` for all burn classes in the active cell.

## 6. Initial Moisture State

Two different initialization ideas exist in the notebook.

### 6.1 Earlier midpoint-style initialization

An earlier node-based initialization computes saturation fraction as:

```math
S_0 = \frac{0.5(\theta_{fc} - \theta_{wp}) + \theta_{wp}}{n}
```

where:

- `\theta_fc` = volumetric field capacity
- `\theta_wp` = volumetric wilting point
- `n` = porosity

### 6.2 Later SMAP-based initialization

A later cell reads an event-start daily SMAP volumetric soil moisture value `\theta_0` and converts it to degree of saturation at cell level:

```math
S_0 = \frac{\theta_0}{n}
```

then clips it:

```math
S_0 = \mathrm{clip}(S_0, 0.05, 0.95)
```

This later cell is the more explicit calibration-oriented initialization.

## 7. Cell-Level Saturation Thresholds

The notebook converts volumetric field capacity and wilting point into degree-of-saturation thresholds:

```math
S_{fc} = \mathrm{clip}\left(\frac{\theta_{fc}}{n}, 0.05, 0.98\right)
```

```math
S_{wp} = \mathrm{clip}\left(\frac{\theta_{wp}}{n}, 0.01, 0.95\right)
```

and enforces:

```math
S_{wp} = \min(S_{wp}, S_{fc} - 0.01)
```

These are stored as cell fields:

- `field_capacity_saturation`
- `wilting_point_saturation`

The custom `SoilMoisture` component imported from `soil_moisture_dynamics1.py` is designed to use these saturation-based thresholds.

## 8. Forcing Construction

### 8.1 CSV forcing path

The notebook contains a cell that reads a daily precipitation and temperature table and distributes temperature using a lapse-rate correction.

Lapse-rate temperature adjustment:

```math
T_{max}(x) = T_{max,ref} - \alpha \frac{z(x)-z_{ref}}{1000}
```

```math
T_{min}(x) = T_{min,ref} - \alpha \frac{z(x)-z_{ref}}{1000}
```

with:

- `\alpha = 4.5` degC/km
- `z_ref = Z_min`

Precipitation is spatially uniform in that block:

```math
P(x) = P_{ref}
```

### 8.2 PRISM forcing path

Later cells read gridded daily PRISM ASC files for:

- precipitation
- minimum temperature
- maximum temperature

For the storm window:

- `2025-12-07` to `2025-12-20`

These PRISM arrays are the ones later used by the hydro-eco loop.

## 9. Snow and Water-Input Module

The notebook builds a simple rain/snow partition and SWE accounting block.

### 9.1 Rain/snow partition

Using thresholds:

- `T_snow = -1.5` degC
- `T_rain = 3.5` degC

Mean daily temperature:

```math
T_{avg} = \frac{T_{min} + T_{max}}{2}
```

Snow fraction:

```math
f_{snow} = \mathrm{clip}\left(\frac{T_{rain} - T_{avg}}{T_{rain} - T_{snow}}, 0, 1\right)
```

Partitioned depths:

```math
P_{snow} = P \times f_{snow}
```

```math
P_{rain} = P \times (1 - f_{snow})
```

### 9.2 SWE and melt bookkeeping

The notebook sets:

- `melt_factor = 1.3` mm/degC/day
- `T_base = 0.0` degC
- `initial_swe_mm = 118.42`

Potential melt:

```math
M_{pot} = m_f \times \max(T_{avg} - T_{base}, 0)
```

Actual melt:

```math
M = \min(SWE_{t-1} + P_{snow}, M_{pot})
```

SWE update:

```math
SWE_t = SWE_{t-1} + P_{snow} - M
```

Water input to the ecohydrology model:

```math
W_{in} = P_{rain} + M
```

Important notebook behavior:

- in the daily ecohydrology loop, the variable named `rainfall` is actually `water_input_arrays`, not raw precipitation.
- therefore the `Precipitation` field passed into the soil-moisture model is already liquid water input after snow partition and melt.

## 10. Daily Hydro-Eco Simulation

The main daily loop iterates over:

- `water_input_arrays`
- `tempmin_arrays`
- `tempmax_arrays`
- `swe_arrays`

For each day it:

1. Adds node fields `Precipitation`, `Tmin`, `Tmax`
2. Maps `Precipitation` to cell `rainfall__daily_depth`
3. Maps hydraulic and vegetation fields to cells
4. Applies a conductivity reduction factor:

```math
K_{sat,cell} = 0.5 \times K_{sat,node\rightarrow cell}
```

5. Updates `Radiation`
6. Updates `PotentialEvapotranspiration`
7. Updates `SoilMoisture`
8. Transfers selected cell outputs back to nodes

### 10.1 Main ecohydrology outputs from `SoilMoisture`

From `soil_moisture_dynamics1.py`, the key outputs are:

- `soil_moisture__root_zone_leakage` [mm]
- `soil_moisture__saturation_fraction` [-]
- `surface__evapotranspiration` [mm]
- `surface__runoff` [mm]
- `vegetation__water_stress` [-]

The notebook stores node-level time stacks:

- `recharge_arrays` from root-zone leakage
- `runoff_arrays`
- `soil_moisture_arrays`
- `ET_arrays`

Then aggregates:

```math
\overline{R}(x) = \text{mean over days of recharge}
```

```math
R_{max}(x) = \text{max over days of recharge}
```

and similarly for runoff.

## 11. Recharge Scenarios

The notebook contains both a routed and a local recharge scenario.

### 11.1 Local recharge scenario

The active landslide run cell defines:

```math
R_{LS}(x) = R_{max}(x)
```

with a floor:

```math
R_{LS}(x) = \max(R_{LS}(x), 0.01)
```

and standard deviation:

```math
\sigma_R(x) = 0.1 \times R_{LS}(x)
```

### 11.2 Routed recharge scenario

The helper function `route_recharge_field` in `notebook/recharge_routing.py` computes an upslope-area-averaged recharge proxy.

It first injects local recharge as:

- `water__unit_flux_in`

Then runs flow accumulation and computes:

```math
R_{routed}(x) = \frac{Q(x)}{A(x)}
```

where:

- `Q(x)` = accumulated `surface_water__discharge`
- `A(x)` = drainage area

The helper also writes:

- `routed_recharge`
- `diff_recharge = routed_recharge - local_recharge`

### 11.3 Important current behavior

The notebook currently contains both scenario branches, but the **active landslide run cell uses local `max_recharge`**, not `routed_recharge_max`.

So the core executed landslide probability block is currently:

```math
R_{LS} = R_{max,local}
```

not:

```math
R_{LS} = R_{max,routed}
```

## 12. Landslide Probability Model

The notebook calls Landlab:

- `LandslideProbability`

with:

- `number_of_iterations = 1000`
- `groundwater__recharge_distribution = "lognormal_spatial"`

### 12.1 Recharge sampling

For each node, the recharge distribution parameters are:

```math
\mu_{\ln R} = \ln\left(\frac{\bar{R}^2}{\sqrt{\sigma_R^2 + \bar{R}^2}}\right)
```

```math
\sigma_{\ln R} = \sqrt{\ln\left(1 + \frac{\sigma_R^2}{\bar{R}^2}\right)}
```

Then:

```math
R_k \sim \mathrm{Lognormal}(\mu_{\ln R}, \sigma_{\ln R})
```

Landlab converts recharge from mm/day to m/day internally.

### 12.2 Sampled strength and geometry terms

For each node, Landlab samples:

```math
C \sim \mathrm{Triangular}(C_{min}, C_{mode}, C_{max})
```

```math
\phi \sim \mathrm{Triangular}(0.82\phi_m, \phi_m, 1.32\phi_m)
```

```math
h_s \sim \mathrm{Triangular}(0.7h_m, h_m, 1.1h_m)
```

Because the notebook provides nonzero `soil__saturated_hydraulic_conductivity`, the Landlab component uses sampled hydraulic conductivity and sampled thickness to compute transmissivity internally:

```math
T = K_{sat} \times h_s
```

rather than sampling transmissivity directly from the raster field.

### 12.3 Slope term used by Landlab

The notebook stores `topographic__slope` as slope gradient. Landlab converts it through:

```math
\beta = \arctan(\theta)
```

where `\theta` is slope gradient.

### 12.4 Relative wetness

Landlab computes relative wetness as:

```math
w = \frac{R}{T} \times \frac{a}{\sin\beta}
```

and caps it at `1`.

### 12.5 Dimensionless cohesion

```math
C^* = \frac{C}{h_s \rho_s g}
```

where:

- `\rho_s` = soil density
- `g` = gravity

### 12.6 Friction / pore-pressure term

```math
Y = \tan\phi \times (1 - 0.5w)
```

The factor `0.5` is the fixed density-ratio term used in the Landlab code.

### 12.7 Factor of safety

The notebook ultimately relies on Landlab’s infinite-slope Monte Carlo formulation:

```math
FS = \frac{C^*}{\sin\beta} + \frac{\cos\beta}{\sin\beta}Y
```

### 12.8 Probability outputs

Probability of saturation:

```math
P_{sat} = \frac{1}{N} \sum_{k=1}^{N} I(w_k \ge 1)
```

Probability of failure:

```math
P_f = \frac{1}{N} \sum_{k=1}^{N} I(FS_k \le 1)
```

with `N = 1000` Monte Carlo samples in the notebook.

## 13. Exported and Derived Outputs

### 13.1 Main model output fields

The notebook creates or inspects:

- `landslide__probability_of_failure`
- `soil__probability_of_saturation`
- `soil__mean_relative_wetness`
- `groundwater__recharge_mean`

### 13.2 Intermediate hydro fields

It also stores:

- `mean_recharge`
- `max_recharge`
- `mean_runoff`
- `groundwater__runoff_mean`
- `routed_recharge_max`
- `distributed_recharge_mean`
- `local_runoff_mean`
- `routed_runoff_m3_day`
- `soil_moisture_mean`
- `ET_mean`

### 13.3 Export formats

The main export block writes, for selected fields:

- GeoTIFF
- ESRI ASCII

to:

```text
/mnt/c/Users/amehedi/Downloads/ml_debris/pioneer/output/cut1
```

## 14. Current Notebook Caveats

These are important if the notebook is used as the source of truth.

### 14.1 Two forcing paradigms coexist

The notebook contains both:

- CSV daily forcing distributed by lapse rate
- gridded PRISM daily ASC forcing

The later hydro run uses the PRISM arrays.

### 14.2 CSV forcing metadata is inconsistent

The cell reads:

- `precip_2026-01.csv`
- `temp_2026-01.csv`

but validates against dates `2025-12-07` to `2025-12-20`, and the error text still mentions `2026-01-01` to `2026-01-31`.

That block should be treated as inconsistent and needs cleanup.

### 14.3 Burn severity currently does not alter cohesion in the active run

Although the notebook defines `post_fire_cohesion`, the current active function returns unity multipliers for burn classes 2, 3, and 4, and the notebook re-adds the original cohesion fields.

So the final landslide run currently uses the original cohesion rasters.

### 14.4 Routed recharge is computed but not used in the active landslide cell

`routed_recharge_max` is created, but the active `LandslideProbability` call later resets recharge to local `max_recharge`.

### 14.5 The notebook contains both production and analysis material

After the main run, the notebook includes:

- SMAP comparison plots
- custom plotting utilities
- recharge/runoff diagnostics
- later refactor-oriented `run_pipeline(...)` cells

These later cells are useful for analysis and migration, but they are not part of the minimum core landslide workflow.

## 15. Suggested Module Boundaries

If this notebook is turned into a cleaner reproducible workflow, the natural modules are:

1. `io_static.py`
   - DEM and static raster loading
2. `terrain.py`
   - slope, drainage area, SCA, boundary handling
3. `forcing.py`
   - daily forcing ingestion and validation
4. `snow.py`
   - rain/snow partition, melt, SWE, water input
5. `ecohydrology.py`
   - PET, radiation, soil-moisture daily loop
6. `recharge.py`
   - local recharge summaries and routed recharge
7. `landslide.py`
   - Monte Carlo landslide probability scenarios
8. `export.py`
   - GeoTIFF/ASC writing
9. `diagnostics.py`
   - SMAP comparison and plots

## 16. Minimal Core Workflow Summary

The notebook’s implemented computational chain is:

```text
DEM + static rasters
-> Landlab terrain fields
-> daily forcing arrays
-> rain/snow partition + SWE + liquid water input
-> daily PET/radiation/soil-moisture updates
-> recharge and runoff time stacks
-> max or routed recharge field
-> LandslideProbability Monte Carlo
-> probability of failure / saturation / wetness outputs
```

That is the core technical workflow currently implemented in the notebook.
