# Spring-Powered Vehicle Drivetrain Optimizer

Physics-based tooling for evaluating spring-powered vehicle drivetrain designs across a configurable sweep of vehicle mass, gear ratio, and wheel diameter for a 40-spring catalog.

The repository currently focuses on the optimization workflow under [`optimization/`](optimization/), which performs a large design-space search, filters infeasible combinations, extracts Pareto-optimal trade-offs, and generates reports and plots.

For implementation details and the optimization theory, see [`optimization/README.md`](optimization/README.md).

## What It Does

- Loads a 40-spring catalog from [`optimization/springs.txt`](optimization/springs.txt)
- Reads physical constants and sweep bounds from [`optimization/config.json`](optimization/config.json)
- Evaluates a configurable Cartesian product of:
  - spring selection
  - vehicle mass
  - gear ratio
  - wheel diameter
- Rejects configurations that fail traction or energy constraints
- Extracts Pareto-optimal designs across:
  - maximum peak speed
  - minimum total mass
  - minimum gear ratio
- Produces CSV summaries, ranked text reports, and plots of the final trade-off surface
- Includes a single-configuration analyzer for inspecting one chosen setup in detail

With the current default `config.json`, the optimizer evaluates more than 69 million parameter combinations before Pareto filtering.

## Core Model

The vehicle is modeled as a torsional spring driving a wheel through a gearbox. The release phase is treated analytically rather than with time stepping, which makes large parameter sweeps practical.

The model accounts for:

- spring mass as part of the accelerated mass
- drivetrain efficiency
- friction-limited traction at the drive wheels
- an energy requirement to reach target speed with a safety factor

The optimization is multi-objective, so the result is not a single "best" design. Instead, the code returns the nondominated trade-off set.

## Main Files

```text
.
├── README.md
└── optimization/
    ├── README.md
    ├── config.json
    ├── explore.py
    ├── spring.py
    ├── spring_model.py
    ├── springs.txt
    ├── graphs/
    └── results/
```

## Workflow

### Full optimization

Run from the repository root:

```bash
python optimization/explore.py
```

Useful options:

```bash
python optimization/explore.py --interactive
python optimization/explore.py --save-all
python optimization/explore.py -f /path/to/custom_springs.txt
```

### Single configuration analysis

```bash
python optimization/spring.py -r 22.5 -d 100 -m 2.06 -s SPF-0927
```

This prints a detailed report for one design and saves a release-response plot in [`optimization/graphs/`](optimization/graphs/).

## Generated Outputs

The optimization workflow writes outputs under [`optimization/results/`](optimization/results/) and [`optimization/graphs/`](optimization/graphs/).

Common artifacts include:

- `pareto_optimal.csv` - final Pareto set
- `key_configurations.txt` - fastest, lightest, simplest, and balanced picks
- `pareto_3d_speed_mass_ratio.png` - 3D objective-space view
- `pareto_2d_projections.png` - 2D trade-off slices
- `pareto_parallel_coordinates.png` - multi-metric comparison plot
- `all_feasible.csv` - optional full feasible set, only written with `--save-all`

## Configuration

[`optimization/config.json`](optimization/config.json) controls:

- target speed
- friction coefficient
- safety factor
- drivetrain efficiency
- wheel-count assumptions
- sweep bounds and step sizes
- plot DPI

Because the sweep is config-driven, result counts change when you change the search bounds or resolution.

## Dependencies

- Python 3
- NumPy
- SciPy
- Matplotlib
- tqdm

## Notes

- Vehicle mass inputs exclude spring mass; the physics model adds spring mass internally when computing total accelerated mass.
- The traction check is evaluated at peak force, which occurs at release start.
- The code is organized so that [`optimization/spring_model.py`](optimization/spring_model.py) contains the reusable physics formulas, while `explore.py` and `spring.py` handle orchestration and reporting.
