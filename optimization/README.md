# Optimization Engine

This directory contains the full drivetrain search and Pareto extraction pipeline for the spring-powered vehicle model. The code evaluates a finite design grid, removes infeasible points, extracts nondominated trade-off solutions, then uses SciPy to refine the search in continuous space.

The main entry points are [`explore.py`](./explore.py), [`spring.py`](./spring.py), and [`spring_model.py`](./spring_model.py). Input spring data comes from [`springs.txt`](./springs.txt), which mirrors the root catalog at [`../springs.txt`](../springs.txt).

## Entry Points

### [`explore.py`](./explore.py)

Runs the full constrained design sweep, filters feasible configurations, extracts the Pareto front, and writes the summary tables and plots.

### [`spring.py`](./spring.py)

Analyzes one specific drivetrain configuration passed on the command line. It uses the shared formulas in [`spring_model.py`](./spring_model.py), prints the derived metrics for that exact setup, and generates the same three-panel release plot for:

- position `x(t)`
- velocity `v(t)`
- acceleration `a(t)`

Usage:

```bash
python spring.py <gear_ratio> <wheel_diameter_mm> <vehicle_mass_kg> <spring_name>
```

Example:

```bash
python spring.py 25 140 1.5 SPF-0927
```

The `vehicle_mass_kg` argument excludes spring mass. The script loads the spring mass from the catalog and computes total accelerated mass as:

```text
m_total = m_vehicle + m_spring
```

All dynamics and constraint checks in `spring.py`, including the wheel-slip / traction-limit check, use `m_total` rather than vehicle mass alone.

The generated plot is written to [`graphs/`](./graphs/). Its filename includes the input vehicle mass only, not total mass.

## What This Optimizer Solves

For each spring in the catalog, the optimizer searches over three continuous design variables:

- vehicle mass `m_v`
- gear ratio `rho`
- wheel radius `R`

The total accelerated mass is always

```text
m = m_v + m_spring
```

and every candidate must satisfy two hard constraints:

```text
peak tractive force <= friction-limited traction
available spring energy >= required kinetic energy * safety factor
```

In both cases, the mass term is the total accelerated mass `m = m_v + m_spring`.

Among feasible points, the optimizer keeps trade-off solutions across three objectives:

1. maximize peak speed
2. minimize total mass
3. minimize gear ratio

This is a multi-objective optimization problem. There is no single best design unless you introduce an extra preference rule. The code therefore returns a Pareto front instead of one scalar optimum.

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

The closed-form structure matters because it makes the expensive parts of the search cheap:

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

The `(pi / 180)` term is required because the catalog angle is in degrees, while energy integrates torque over radians.

### Pareto Optimality

The optimizer uses the standard dominance relation from multi-objective optimization.

Given two feasible objective vectors `a` and `b`, where all objectives are written as quantities to minimize, `a` dominates `b` if:

```text
for every objective i:   a_i <= b_i
for at least one j:      a_j <  b_j
```

That means `a` is no worse in every objective and strictly better in at least one.

A point is Pareto-optimal if no other feasible point dominates it. The set of Pareto-optimal decision vectors is the Pareto set. Their image in objective space is the Pareto front.

This is a partial order, not a total order. Two designs are often incomparable:

- one may be lighter
- the other may be faster

Neither dominates the other, so both remain on the front.

In this codebase the objectives are encoded as:

```text
[-speed, total_mass, gear_ratio]
```

because the Pareto extractor assumes minimization. Negating speed converts "maximize speed" into a minimization problem.

### Why Pareto Instead of a Weighted Score

A weighted score like

```text
J = w1 * (-speed) + w2 * mass + w3 * ratio
```

would force one preference system into the search. That is useful if you already know the exact business trade-off, but it hides alternatives. Pareto filtering preserves the full set of nondominated choices so you can decide later whether fast, light, or mechanically simple matters more.

### Differential Evolution in SciPy

After the coarse grid sweep, [`explore.py`](./explore.py) refines solutions with [`scipy.optimize.differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

Differential evolution is a population-based global optimizer for continuous variables. At each generation it keeps a population of candidate vectors and builds new trial vectors from differences between existing ones.

For SciPy's default `best1bin` strategy, a mutant is formed as:

```text
v_i = x_best + F * (x_r1 - x_r2)
```

where:

- `x_best` is the best current individual
- `x_r1`, `x_r2` are distinct random individuals
- `F` is the mutation factor

Then binomial crossover mixes the target vector `x_i` and mutant `v_i` componentwise to form a trial vector `u_i`:

```text
u_i[j] = v_i[j]  if rand_j <= CR or j == j_rand
u_i[j] = x_i[j]  otherwise
```

Finally, selection is greedy:

```text
x_i(next) = u_i     if u_i is better
x_i(next) = x_i     otherwise
```

For constrained problems, SciPy applies a constraint-aware comparison so feasible points are preferred over infeasible ones.

This method fits the repository well because the search space is:

- continuous in three dimensions
- nonlinear
- constrained
- not something we want to differentiate by hand

The code also sets `polish=True`. In SciPy that means the best final differential-evolution candidate is passed to `scipy.optimize.minimize` for a local refinement step. For constrained problems, SciPy uses `trust-constr` rather than `L-BFGS-B`.

## How Execution Actually Proceeds

### 1. Argument parsing and output setup

[`explore.py`](./explore.py) parses the spring file path and the optional interactive plotting flag, then creates [`results/`](./results/) and [`graphs/`](./graphs/).

```python
parser = argparse.ArgumentParser(
    description="Multi-objective optimization of spring-powered vehicle drivetrain"
)
parser.add_argument("-i", "--interactive", action="store_true")
parser.add_argument("-f", "--file", default="springs.txt", metavar="SPRINGS_FILE")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
```

### 2. Spring catalog is normalized into SI units

[`spring_model.py`](./spring_model.py) parses the tab-separated catalog and converts:

- `Nmm -> Nm`
- `g -> kg`

```python
entry = {
    "max_rotation_deg": float(parts[1]),
    "max_torque_nm": float(parts[2]) / 1000.0,
}
entry["mass_kg"] = float(parts[3]) / 1000.0
```

This is important because every later equation assumes SI units internally.

### 3. The coarse search is built as a 4D vectorized grid

The search space is the Cartesian product of:

- spring index
- vehicle mass values
- gear ratio values
- wheel diameter values

Instead of nested loops, the code reshapes one-dimensional arrays so NumPy broadcasting can compute every metric in one pass.

```python
torque_4d = torque_arr[:, None, None, None]
theta0_4d = theta0_arr[:, None, None, None]
spr_m_4d = spr_mass_arr[:, None, None, None]

veh_m_4d = veh_mass_vals[None, :, None, None]
rho_4d = ratio_vals[None, None, :, None]
R_4d = diam_vals[None, None, None, :] / 2000.0

total_m_4d = veh_m_4d + spr_m_4d
```

This creates arrays with shape `(n_springs, n_vehicle_masses, n_ratios, n_diameters)`. In the current configuration that is:

```text
40 * 21 * 29 * 31 = 755,160
```

The asymptotic work is still linear in the number of grid points, but vectorization moves the inner loops into compiled NumPy code and removes Python loop overhead.

### 4. Feasibility is checked before Pareto filtering

The script computes the full grid of forces, energies, speed, target time, target distance, traction margin, and initial acceleration. Then it discards any point that violates traction or energy.

```python
avail_E = DRIVETRAIN_EFF * E_st_4d
req_E = 0.5 * total_m_4d * TARGET_SPEED_MS ** 2 * SAFETY_FACTOR

peak_F = sm.max_force_ground(torque_4d, rho_4d, R_4d, DRIVETRAIN_EFF)
trac_L = sm.traction_limit(total_m_4d, GRAVITY, N_DRIVING_WHEELS,
                           N_WHEELS, FRICTION_COEFF, SAFETY_FACTOR)

feasible = (peak_F <= trac_L) & (avail_E >= req_E)
```

This matters mathematically because Pareto comparisons only make sense within the feasible set. An infeasible design cannot dominate a feasible one in the final engineering decision.

### 5. Feasible points are flattened into a tabular dataset

The code broadcasts parameter arrays to the same 4D shape, ravels them, and applies the feasibility mask. That produces aligned one-dimensional arrays that are easy to save and compare.

```python
mask = feasible.ravel()
F_s = s_flat[mask].copy()
F_vm = vm_flat[mask].copy()
F_tm = tm_flat[mask].copy()
F_rho = rho_flat[mask].copy()
F_spd = spd_flat[mask].copy()
```

The resulting table is written to [`results/all_feasible.csv`](./results/all_feasible.csv).

### 6. The first Pareto front is extracted from the feasible grid

The core extractor is `pareto_front_indices(costs)` in [`explore.py`](./explore.py).

```python
grid_costs = np.column_stack([-F_spd, F_tm, F_rho])
grid_pareto_idx = pareto_front_indices(grid_costs)
```

Inside the extractor, the algorithm:

1. normalizes each objective range
2. sorts points by the sum of normalized costs so likely-good candidates are visited early
3. rejects any point dominated by the current Pareto set
4. removes any existing Pareto member dominated by the new point

```python
if np.any(np.all(pc <= c, axis=1) & np.any(pc < c, axis=1)):
    continue

keep = ~(np.all(c <= pc, axis=1) & np.any(c < pc, axis=1))
```

This is not a full divide-and-conquer Pareto solver. It is an incremental nondominance filter with a heuristic ordering step. In the worst case, nondominated filtering is still quadratic in the number of points, `O(N^2 * k)`, where `N` is the number of feasible points and `k` is the number of objectives. The sort improves practical behavior because dominated points are often removed quickly.

### 7. SciPy refinement is run per spring and per objective

The second phase solves a constrained continuous optimization problem for each spring. The decision vector is:

```text
x = [vehicle_mass, gear_ratio, wheel_radius]
```

and the bounds are:

```python
OPT_BOUNDS = [
    (LOWER_BOUND_MASS, UPPER_BOUND_MASS),
    (LOWER_BOUND_RATIO, UPPER_BOUND_RATIO),
    (LOWER_BOUND_DIAM / 2000.0, UPPER_BOUND_DIAM / 2000.0),
]
```

Two nonlinear constraints are defined as signed margins. A point is feasible when both are nonnegative:

```python
def _noslip(x, _tau=tau, _ms=m_spr):
    mt = x[0] + _ms
    return float(
        sm.traction_limit(mt, GRAVITY, N_DRIVING_WHEELS,
                          N_WHEELS, FRICTION_COEFF, SAFETY_FACTOR)
        - sm.max_force_ground(_tau, x[1], x[2], DRIVETRAIN_EFF))

def _energy(x, _E=E_spr, _ms=m_spr):
    mt = x[0] + _ms
    return float(DRIVETRAIN_EFF * _E
                 - 0.5 * mt * TARGET_SPEED_MS ** 2 * SAFETY_FACTOR)
```

Those functions are wrapped by [`NonlinearConstraint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html):

```python
cons = [
    NonlinearConstraint(_noslip, 0, np.inf),
    NonlinearConstraint(_energy, 0, np.inf),
]
```

For each spring, the optimizer is run three times:

- once to maximize speed
- once to minimize total mass
- once to minimize gear ratio

Speed is negated to convert maximization into minimization:

```python
def _obj_speed(x, _th0=th0, _kth=kth, _ms=m_spr):
    mt = x[0] + _ms
    return float(-sm.max_speed(_th0, mt, _kth, DRIVETRAIN_EFF) * 3.6)
```

Then SciPy is called:

```python
res = differential_evolution(
    obj_fn,
    OPT_BOUNDS,
    constraints=cons,
    seed=42,
    maxiter=200,
    tol=1e-8,
    polish=True,
)
```

Conceptually, this phase fills the gaps left by the finite grid. The grid may miss a good point that lies between sampled masses, ratios, or diameters. Differential evolution searches the continuous box instead of the sampled lattice.

### 8. Grid points and refined points are merged, then Pareto-filtered again

Refined candidates are appended to the feasible grid arrays, and the Pareto extraction step is repeated on the combined set.

```python
all_costs = np.column_stack([-A_spd, A_tm, A_rho])
pareto_idx = pareto_front_indices(all_costs)
```

This second pass is important. A refined point may dominate a grid point, and a grid point may still remain relevant if SciPy does not improve that region.

### 9. Reporting and visualization are generated from the final Pareto set

The final outputs are written to:

- [`results/all_feasible.csv`](./results/all_feasible.csv)
- [`results/pareto_optimal.csv`](./results/pareto_optimal.csv)
- [`results/key_configurations.txt`](./results/key_configurations.txt)
- [`graphs/pareto_3d_speed_mass_ratio.png`](./graphs/pareto_3d_speed_mass_ratio.png)
- [`graphs/pareto_2d_projections.png`](./graphs/pareto_2d_projections.png)
- [`graphs/pareto_parallel_coordinates.png`](./graphs/pareto_parallel_coordinates.png)

The "balanced" pick in [`results/key_configurations.txt`](./results/key_configurations.txt) is not another Pareto algorithm. It is a post-processing choice that normalizes the three Pareto objectives to `[0, 1]` and picks the point maximizing:

```text
normalized_speed - normalized_mass - normalized_ratio
```

That gives one interpretable compromise design without changing the actual Pareto front.

## Code Map

### [`spring_model.py`](./spring_model.py)

Pure physics functions and unit-normalized data loading.

- `load_springs()` parses the catalog
- `stored_energy()` computes spring energy
- `omega()`, `release_time()`, `max_speed()` implement the closed-form kinematics
- `position()`, `velocity()`, and `acceleration()` generate the release-response curves
- `time_to_speed()` and `distance_to_speed()` compute target-reaching metrics
- `max_force_ground()` and `traction_limit()` implement the hard constraints

This separation keeps the model reusable and makes [`explore.py`](./explore.py) and [`spring.py`](./spring.py) mostly orchestration scripts.

### [`spring.py`](./spring.py)

Single-configuration analyzer.

- parses `gear_ratio`, `wheel_diameter_mm`, `vehicle_mass_kg`, and `spring_name`
- loads physics constants from [`config.json`](./config.json)
- loads the spring catalog from [`springs.txt`](./springs.txt)
- computes force, energy, speed, timing, and traction metrics for one configuration
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

There are three design choices worth calling out.

### Closed-form physics instead of time stepping

Because the release model is analytically solvable, the code avoids numerical integration. That removes solver error, reduces runtime, and makes the optimization stable. For this problem, direct formulas are simpler and more accurate than integrating an ODE at every candidate point.

### Grid search plus global refinement

The grid gives broad coverage and easy reporting. Differential evolution then improves local resolution without requiring a dense grid everywhere. In practice this is a good fit for a three-variable engineering search because it combines explainability with continuous refinement.

### Pareto front instead of a single objective scalarization

The project is meant to expose the trade-off surface, not hide it. A speed-focused design, a lightweight design, and a simple drivetrain design are all valid engineering answers depending on downstream priorities. The Pareto front keeps those alternatives visible.

## References

- SciPy differential evolution: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>
- SciPy nonlinear constraints: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html>
- NumPy broadcasting: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- Pareto optimality overview: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6105305/>
- Multi-objective optimization overview: <https://en.wikipedia.org/wiki/Multi-objective_optimization>
- Differential evolution survey: <https://www.math.ucdavis.edu/~saito/data/PSO-ACO/opara-arabas_differential-evol-survey.pdf>
