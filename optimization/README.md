# Optimization Engine

This directory contains the active drivetrain search and reporting workflow for the spring-powered vehicle model. It evaluates a configurable design grid, filters infeasible points, extracts Pareto-optimal trade-offs, and refines the search in continuous space with SciPy.

The main entry points are [`explore.py`](./explore.py), [`spring.py`](./spring.py), and [`spring_model.py`](./spring_model.py). Input spring data lives in [`springs.txt`](./springs.txt), and physical constants plus sweep bounds live in [`config.json`](./config.json).

## Entry Points

### [`explore.py`](./explore.py)

Runs the full constrained design sweep, filters feasible configurations, extracts the Pareto front, refines candidates with differential evolution, and writes summary tables and plots.

Usage:

```bash
python explore.py [-i] [-f SPRINGS_FILE] [--save-all]
```

Notes:

- `-i` or `--interactive` shows plots interactively while saving them
- `-f` or `--file` loads a different spring catalog
- `--save-all` writes [`results/all_feasible.csv`](./results/all_feasible.csv) for every feasible configuration

### [`spring.py`](./spring.py)

Analyzes one specific drivetrain configuration passed on the command line. It uses the shared formulas in [`spring_model.py`](./spring_model.py), prints the derived metrics for that setup, and generates a three-panel release plot for:

- position `x(t)`
- velocity `v(t)`
- acceleration `a(t)`

Usage:

```bash
python spring.py -r <ratio> -d <diameter_mm> -m <vehicle_mass_kg> -s <spring>
```

Example:

```bash
python spring.py -r 22.5 -d 100 -m 2.06048 -s SPF-0927
```

The `vehicle_mass_kg` argument excludes spring mass. The script loads spring mass from the catalog and computes total accelerated mass as:

```text
m_total = m_vehicle + m_spring
```

All dynamics and constraint checks in `spring.py`, including the wheel-slip limit, use `m_total` rather than vehicle mass alone.

## What This Optimizer Solves

For each spring in the catalog, the optimizer searches over three design variables:

- vehicle mass `m_v`
- gear ratio `rho`
- wheel radius `R`

The total accelerated mass is always:

```text
m = m_v + m_spring
```

and every candidate must satisfy two hard constraints:

```text
peak tractive force <= friction-limited traction
available spring energy >= required kinetic energy * safety factor
```

Among feasible points, the optimizer keeps trade-off solutions across three objectives:

1. maximize peak speed
2. minimize total mass
3. minimize gear ratio

This is a multi-objective optimization problem, so the code returns a Pareto front instead of a single scalar optimum.

## Configuration-Driven Sweep Size

[`config.json`](./config.json) defines the sweep bounds and step sizes. The grid size is:

```text
number of springs * number of mass samples * number of ratio samples * number of diameter samples
```

With the current default config:

- 40 springs
- masses from 2.0 to 5.0 kg in 0.05 kg steps: 61 samples
- ratios from 10.0 to 80.0 in 0.5 steps: 141 samples
- diameters from 50.0 to 150.0 mm in 0.5 mm steps: 201 samples

which yields:

```text
40 * 61 * 141 * 201 = 69,152,040 combinations
```

That number changes whenever `config.json` changes.

## Theory

### Physics Model

The release model in [`spring_model.py`](./spring_model.py) treats the spring-powered drivetrain as an ideal torsional spring driving a wheel through a gearbox. The motion is simple harmonic during the release phase:

```text
theta'' + omega^2 theta = 0
```

with

```text
omega = sqrt(180 * eta * k_theta / (pi * m * rho^2 * R^2))
```

where:

- `eta` is drivetrain efficiency
- `k_theta` is spring rate in `Nm/deg`
- `m` is total mass
- `rho` is gear ratio
- `R` is wheel radius

This closed-form structure makes the search practical because:

- peak speed can be computed directly from energy conservation
- target time and target distance can be computed directly from inverse trigonometric expressions
- peak tractive force occurs at `t = 0`, so the no-slip check is a single inequality instead of a time simulation

The key energy relation is:

```text
v_max = sqrt(2 * eta * E / m)
```

and the stored spring energy is:

```text
E = 0.5 * k_theta * theta_0^2 * (pi / 180)
```

The `(pi / 180)` factor is required because the catalog angle is stored in degrees, while energy integrates torque over radians.

### Pareto Optimality

The optimizer uses the standard dominance relation from multi-objective optimization.

Given two feasible objective vectors `a` and `b`, where all objectives are written as quantities to minimize, `a` dominates `b` if:

```text
for every objective i:   a_i <= b_i
for at least one j:      a_j <  b_j
```

In this codebase the objectives are encoded as:

```text
[-speed, total_mass, gear_ratio]
```

because the Pareto extractor assumes minimization. Negating speed converts "maximize speed" into a minimization problem.

### Differential Evolution in SciPy

After the coarse grid sweep, [`explore.py`](./explore.py) refines solutions with [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

Differential evolution is a population-based global optimizer for continuous variables. It fits this repository well because the search space is:

- continuous in three dimensions
- nonlinear
- constrained
- not something we want to differentiate by hand

The code also sets `polish=True`, so SciPy locally refines the best final differential-evolution candidate.

## How Execution Proceeds

### 1. Argument parsing and output setup

[`explore.py`](./explore.py) parses CLI options, loads [`config.json`](./config.json), and creates [`results/`](./results/) plus [`graphs/`](./graphs/).

### 2. Spring catalog is normalized into SI units

[`spring_model.py`](./spring_model.py) parses the tab-separated catalog and converts:

- `Nmm -> Nm`
- `g -> kg`

### 3. The coarse search is built as a 4D vectorized grid

The search space is the Cartesian product of:

- spring index
- vehicle mass values
- gear ratio values
- wheel diameter values

NumPy broadcasting computes every metric in one pass instead of nested Python loops.

### 4. Feasibility is checked before Pareto filtering

The script computes force, energy, speed, target time, target distance, traction margin, and initial acceleration, then discards any point that violates traction or energy.

### 5. Feasible points are flattened into a tabular dataset

Aligned arrays are flattened and filtered by the feasibility mask so they can be saved and compared directly.

[`results/all_feasible.csv`](./results/all_feasible.csv) is only written when `--save-all` is supplied.

### 6. The first Pareto front is extracted from the feasible grid

The grid candidates are converted into the cost vector:

```text
[-speed, total_mass, gear_ratio]
```

and filtered for nondominance.

### 7. SciPy refinement is run per spring and per objective

For each spring, differential evolution is run separately to:

- maximize speed
- minimize total mass
- minimize gear ratio

subject to the traction and energy constraints.

### 8. Grid points and refined points are merged, then Pareto-filtered again

Refined candidates are appended to the feasible grid arrays, and the Pareto extraction step is repeated on the combined set.

### 9. Reporting and visualization are generated from the final Pareto set

The final outputs are written to:

- [`results/pareto_optimal.csv`](./results/pareto_optimal.csv)
- [`results/key_configurations.txt`](./results/key_configurations.txt)
- [`graphs/pareto_3d_speed_mass_ratio.png`](./graphs/pareto_3d_speed_mass_ratio.png)
- [`graphs/pareto_2d_projections.png`](./graphs/pareto_2d_projections.png)
- [`graphs/pareto_parallel_coordinates.png`](./graphs/pareto_parallel_coordinates.png)

Optional or situational outputs include:

- [`results/all_feasible.csv`](./results/all_feasible.csv) when `--save-all` is used
- configuration-specific plots written by [`spring.py`](./spring.py)

The "balanced" pick in [`results/key_configurations.txt`](./results/key_configurations.txt) is a post-processing choice that normalizes the three Pareto objectives to `[0, 1]` and picks the point maximizing:

```text
normalized_speed - normalized_mass - normalized_ratio
```

## Code Map

### [`spring_model.py`](./spring_model.py)

Pure physics functions and unit-normalized data loading.

- `load_springs()` parses the catalog
- `stored_energy()` computes spring energy
- `omega()`, `release_time()`, `max_speed()` implement the closed-form kinematics
- `position()`, `velocity()`, and `acceleration()` generate release-response curves
- `time_to_speed()` and `distance_to_speed()` compute target-reaching metrics
- `max_force_ground()` and `traction_limit()` implement the hard constraints

### [`spring.py`](./spring.py)

Single-configuration analyzer.

- parses `gear_ratio`, `wheel_diameter_mm`, `vehicle_mass_kg`, and `spring_name`
- loads physics constants from [`config.json`](./config.json)
- loads the spring catalog from [`springs.txt`](./springs.txt) by default, or a custom file via `-f`
- computes force, energy, speed, timing, traction, and spring-turn metrics for one configuration
- generates the three-panel `x(t)`, `v(t)`, `a(t)` plot
- saves the plot under [`graphs/`](./graphs/)

### [`explore.py`](./explore.py)

Pipeline coordinator.

- parses arguments
- builds the parameter grid
- computes vectorized metrics
- filters infeasible points
- extracts the first Pareto front
- runs SciPy refinement
- merges candidates and extracts the final Pareto front
- writes CSV and text outputs
- builds plots

## Why The Design Looks Like This

### Closed-form physics instead of time stepping

Because the release model is analytically solvable, the code avoids numerical integration. That removes solver error, reduces runtime, and makes the optimization stable.

### Grid search plus global refinement

The grid gives broad coverage and easy reporting. Differential evolution then improves local resolution without requiring a dense grid everywhere.

### Pareto front instead of a single weighted score

The project is meant to expose the trade-off surface, not hide it. A speed-focused design, a lightweight design, and a simple drivetrain design can all be valid engineering answers depending on downstream priorities.

## References

- SciPy differential evolution: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
- SciPy nonlinear constraints: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html>
- NumPy broadcasting: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- Pareto optimality overview: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6105305/>
- Multi-objective optimization overview: <https://en.wikipedia.org/wiki/Multi-objective_optimization>
