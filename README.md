# Spring-Powered Vehicle Drivetrain Optimizer

Multi-objective optimization toolkit for designing spring-powered vehicle drivetrains. Evaluates 755,160 parameter combinations across a 40-spring catalog to identify Pareto-optimal configurations that balance speed, mass, and mechanical simplicity.

The optimization internals are documented separately in [`optimization/README.md`](optimization/README.md), including the Pareto theory, the differential-evolution refinement step, and how those ideas map to the implementation.

## Description

This project solves the drivetrain design problem for a three-wheeled vehicle powered by a torsional spring. The optimizer explores combinations of:

- **Spring selection** — 40 catalog options (106g–1547g, 0.28–104.5 J stored energy)
- **Vehicle mass** — 1.5–4.0 kg (excluding spring)
- **Gear ratio** — 10–80 (dimensionless)
- **Wheel diameter** — 50–200 mm

The system computes Pareto-optimal solutions across three competing objectives:

1. **Maximize peak speed** (km/h)
2. **Minimize total mass** (vehicle + spring, kg)
3. **Minimize gear ratio** (lower = simpler mechanism)

Hard constraints ensure (1) no wheel slip at peak torque and (2) sufficient energy to reach 15 km/h with a 1.5× safety factor. The optimizer accounts for 92% drivetrain efficiency and includes spring mass in all dynamics calculations. In particular, the slip / traction check uses total mass `(vehicle + spring)`, not vehicle mass alone.

Typical Pareto fronts contain 70 configurations ranging from lightweight/slow designs (1.61 kg, 18.8 km/h, ratio 17.5) to heavy/fast setups (3.05 kg, 49.1 km/h, ratio 62.5).

## Core Techniques & Concepts

### Simple Harmonic Motion Model

The release phase follows ideal spring-mass dynamics. When the torsional spring (rate _k<sub>θ</sub>_ in Nm/deg) unwinds through a gearbox of ratio _ρ_, the vehicle motion satisfies:

```
θ̈ + ω² θ = 0
```

where the angular frequency is:

```
ω = √(180 η k_θ / (π m ρ² R²))
```

- **η** = drivetrain efficiency (0.92)
- **m** = total mass (vehicle + spring, kg)
- **R** = wheel radius (m)
- **ρ** = gear ratio (dimensionless)

The solution is `θ(t) = θ₀ cos(ωt)`, which yields closed-form expressions for velocity, distance, and time-to-target. Peak force occurs at _t_ = 0; if the configuration satisfies the no-slip constraint at _t_ = 0, it remains valid throughout the release.

**Mathematical insight**: Peak speed is independent of gearing and determined solely by energy-to-mass ratio:

```
v_max = √(2 η E / m)
```

However, time and distance to target **do** depend on gearing, creating trade-offs between acceleration, speed capability, and mechanical complexity.

### Energy Calculation

Stored energy integrates torque over angular displacement. Since the catalog provides torque in Nmm and rotation in degrees:

```
E = ½ k_θ θ₀² · (π/180)
```

The `(π/180)` factor converts degree-based spring constant to radian-based energy. Early versions of the code omitted this factor, producing incorrect energy estimates.

### Multi-Objective Pareto Optimization

A configuration **A** dominates **B** if **A** is strictly better in at least one objective and not worse in any other. The Pareto front contains all non-dominated solutions.

**Grid-then-refine strategy**:

1. **Coarse sweep** — 755,160 combinations evaluated with vectorized NumPy operations (21 vehicle masses × 29 ratios × 31 diameters × 40 springs)
2. **Pareto extraction** — O(_n_ log _n_) sort-based algorithm removes dominated points from 107k feasible configs, yielding ~70 Pareto members
3. **Continuous refinement** — [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) with nonlinear constraints refines each spring's best solutions for speed, mass, and ratio objectives independently

Differential evolution is a stochastic global optimizer that maintains a population of candidate solutions and evolves them through mutation and crossover. It handles nonlinear constraints (traction, energy) without requiring gradient information.

### Constraint Enforcement

**Traction constraint** — Peak ground force must not exceed friction limit at the two driving wheels:

```
η (τ_max / ρ) / R  ≤  (μ / SF) · (2/3) · m · g
```

- **μ** = 0.7 (rubber-on-ground friction coefficient)
- **SF** = 1.5 (safety factor)
- **2/3** accounts for weight distribution over 3 wheels, 2 driving

**Energy constraint** — Spring must store enough energy to accelerate the total mass (vehicle + spring) to 15 km/h:

```
η · E ≥ ½ · m · v_target² · SF
```

Both constraints are evaluated using [`NonlinearConstraint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html) in scipy's optimizer.

### Vectorized Grid Computation

The coarse sweep uses [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) to avoid nested loops. Parameter arrays are reshaped to 4D:

```python
torque_4d   = torque_arr  [:, None, None, None]  # (40, 1, 1, 1)
theta0_4d   = theta0_arr  [:, None, None, None]
k_theta_4d  = k_theta_arr [:, None, None, None]
spr_mass_4d = spr_mass_arr[:, None, None, None]

veh_mass_4d = veh_mass_vals   [None, :, None, None]  # (1, 21, 1, 1)
ratio_4d    = ratio_vals       [None, None, :, None]  # (1, 1, 29, 1)
radius_4d   = (diam_vals / 2000.0)[None, None, None, :]  # (1, 1, 1, 31)

total_mass_4d = veh_mass_4d + spr_mass_4d  # (40, 21, 29, 31)
```

Broadcasting automatically computes all combinations in a single vectorized expression. The result is a 755,160-element array of speeds, forces, energies, etc. Time complexity: **O(n)** where _n_ = grid size (versus **O(n)** with four nested loops, but with ~100× faster execution due to C-level NumPy operations).

### Visualization: Parallel Coordinates

The [`pareto_parallel_coordinates.png`](optimization/graphs/pareto_parallel_coordinates.png) plot uses parallel axes to visualize high-dimensional trade-offs. Each polyline represents one Pareto configuration, spanning 10 metrics (spring mass, vehicle mass, total mass, ratio, diameter, speed, time, distance, traction margin, acceleration). The technique reveals correlations (e.g., higher ratios correlate with faster speeds) and constraint boundaries (traction margin never negative).

Lines are color-coded by spring part number, with the colorbar spanning only springs present on the Pareto front (not all 40 catalog entries).

## Non-Obvious Technologies

### scipy.optimize.differential_evolution

Global optimization algorithm from the [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) module. Unlike gradient-based methods (L-BFGS, conjugate gradient), differential evolution is a population-based genetic algorithm that:

- Handles non-convex, multimodal objective landscapes
- Supports nonlinear inequality constraints
- Does not require gradient information (suitable for complex physics models)
- Provides `polish=True` option for final refinement via L-BFGS-B

Used here to refine grid-based Pareto candidates into continuous-domain optima. Each of the 40 springs is optimized independently for three objectives (speed, mass, ratio), totaling 120 optimization runs.

### matplotlib 3D projections

[`mpl_toolkits.mplot3d.Axes3D`](https://matplotlib.org/stable/api/toolkits/mplot3d.html) renders the three-objective Pareto surface in [`pareto_3d_speed_mass_ratio.png`](optimization/graphs/pareto_3d_speed_mass_ratio.png). The projection uses orthographic viewing to avoid perspective distortion of trade-off geometry.

### Bash strict mode

The batch script [`single/run_all_springs.sh`](single/run_all_springs.sh) uses `set -euo pipefail`:

- `-e` — exit immediately if any command fails
- `-u` — treat unset variables as errors
- `-o pipefail` — return pipeline failure status from first failed command

This ensures robust automation when processing all 40 springs in sequence.

## External Links

### Python Libraries

- **NumPy** — [numpy.org](https://numpy.org/)
- **SciPy** — [scipy.org](https://scipy.org/)
- **Matplotlib** — [matplotlib.org](https://matplotlib.org/)

### Spring Catalog Data

The [`springs.txt`](springs.txt) file contains empirical data for 40 commercial torsional springs (part numbers SPF-0900 through SPF-0939). Each entry specifies:

- Maximum rotation angle for 10,000 cycles (degrees)
- Maximum torque for 10,000 cycles (Nmm)
- Spring mass (grams)

Data format: tab-separated values with header row.

## Project Structure

```
.
├── optimization/
├── single/
└── springs.txt
```

### `optimization/`

Multi-objective Pareto optimization of the full parameter space. Contains:

- [`spring_model.py`](optimization/spring_model.py) — Shared physics library with pure functions for energy, kinematics, constraints (no hardcoded defaults)
- [`explore.py`](optimization/explore.py) — Main optimizer; performs 755k-point grid sweep, Pareto extraction, and scipy refinement
- `results/` — Generated outputs:
  - `all_feasible.csv` — 107,832 constraint-satisfying configurations (10 columns each)
  - `pareto_optimal.csv` — 70 Pareto-optimal configurations
  - `key_configurations.txt` — Fastest, lightest, simplest, and balanced designs with top-5 rankings
- `graphs/` — High-resolution (600 DPI) visualizations:
  - `pareto_3d_speed_mass_ratio.png` — 3D scatter of objectives
  - `pareto_2d_projections.png` — Three 2D slices (speed vs. mass, speed vs. ratio, mass vs. ratio)
  - `pareto_parallel_coordinates.png` — 10-metric parallel-axis plot
  - Per-configuration plots produced by `optimization/spring.py` use the input vehicle mass in the filename (for example `mass-1.5kg_...`), even though the plotted dynamics and constraint calculations use total mass `(vehicle + spring)` internally

### `single/`

Per-spring analysis tools for exploring individual spring behavior. Contains:

- [`spring.py`](single/spring.py) — Interactive script for single-spring analysis (e.g., `python spring.py spf-0927`)
- [`run_all_springs.sh`](single/run_all_springs.sh) — Bash automation to run `spring.py` for all 40 catalog entries
- `graphs/` — Generated 3D traction surfaces and time-domain release response plots
- `outputs/` — Text summaries of single-spring analyses (40 files, one per spring)

### `springs.txt`

Master spring catalog (40 entries, 4 columns: part number, rotation, torque, mass). Copied into both `optimization/` and `single/` subdirectories for modular operation.

---

## Key Constants

Default physical parameters used throughout the codebase:

| Parameter               | Value      | Symbol        |
|-------------------------|------------|---------------|
| Target speed            | 15 km/h    | —             |
| Friction coefficient    | 0.7        | μ             |
| Safety factor           | 1.5        | SF            |
| Drivetrain efficiency   | 92%        | η             |
| Wheels (total)          | 3          | —             |
| Driving wheels          | 2          | —             |
| Gravity                 | 9.81 m/s²  | g             |

Sweep ranges (optimization):

- Vehicle mass: 1.5–4.0 kg (step 0.125)
- Gear ratio: 10–80 (step 2.5)
- Wheel diameter: 50–200 mm (step 5)

---

**Note**: All unit conversions (Nmm → Nm, g → kg, mm → m, km/h ↔ m/s) are handled explicitly in the code. Spring mass is included in all dynamics calculations (force, energy, traction).
