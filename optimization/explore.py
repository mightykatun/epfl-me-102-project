#!/usr/bin/env python
"""
Sweep and Pareto-optimise spring-powered vehicle drivetrain parameters.

Explores combinations of spring, vehicle mass, gear ratio, and wheel diameter.
The total accelerated mass is  vehicle_mass + spring_mass.

Extracts the global Pareto front across three objectives:
  - Maximise peak speed
  - Minimise total mass (vehicle + spring)
  - Minimise gear ratio (simpler mechanism)

Hard constraints (both must hold):
  1. No wheel slip at peak force (t = 0)
  2. Sufficient spring energy to reach target speed (with safety factor)

Usage:
    python explore.py [-i] [-f SPRINGS_FILE]
"""

import os
import csv
import json
import warnings
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

warnings.filterwarnings("ignore", message="delta_grad == 0.0")
warnings.filterwarnings("ignore", message=".*tight_layout.*")

from scipy.optimize import differential_evolution, NonlinearConstraint
from tqdm import tqdm

import spring_model as sm

# ═══════════════════════════════════════════════════════════════════════════
# Parse command-line arguments
# ═══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Multi-objective optimization of spring-powered vehicle drivetrain"
)
parser.add_argument(
    "-i", "--interactive",
    action="store_true",
    help="Show plots interactively as they are being saved (default: False)"
)
parser.add_argument(
    "-f", "--file",
    default="springs.txt",
    metavar="SPRINGS_FILE",
    help="Spring data file to load (default: springs.txt)"
)
parser.add_argument(
    "--save-all",
    action="store_true",
    help="Write all_feasible.csv with every feasible configuration (default: False)"
)
args = parser.parse_args()

# Set matplotlib backend based on interactive flag
if not args.interactive:
    matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════════
# Output directories
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
GRAPHS_DIR  = os.path.join(SCRIPT_DIR, "graphs")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

with open(CONFIG_FILE) as fh:
    config = json.load(fh)

physics_config = config["physics"]
sweep_config = config["sweep"]

# ═══════════════════════════════════════════════════════════════════════════
# Constants  (same values as spring.py)
# ═══════════════════════════════════════════════════════════════════════════

TARGET_SPEED_KMH = physics_config["target_speed_kmh"]
TARGET_SPEED_MS  = TARGET_SPEED_KMH / 3.6
FRICTION_COEFF   = physics_config["friction_coeff"]
SAFETY_FACTOR    = physics_config["safety_factor"]
DRIVETRAIN_EFF   = physics_config["drivetrain_eff"]
N_WHEELS         = physics_config["n_wheels"]
N_DRIVING_WHEELS = physics_config["n_driving_wheels"]
GRAVITY          = physics_config["gravity"]
DPI              = physics_config["dpi"]

# ═══════════════════════════════════════════════════════════════════════════
# Load spring catalogue
# ═══════════════════════════════════════════════════════════════════════════

# Use absolute path if provided, otherwise relative to script directory
if os.path.isabs(args.file):
    SPRINGS_FILE = args.file
else:
    SPRINGS_FILE = os.path.join(SCRIPT_DIR, args.file)

springs_data = sm.load_springs(SPRINGS_FILE)
spring_names = sorted(springs_data.keys())
N_SPRINGS    = len(spring_names)

# ═══════════════════════════════════════════════════════════════════════════
# Sweep ranges
# ═══════════════════════════════════════════════════════════════════════════

# kg not counting spring mass
LOWER_BOUND_MASS  = sweep_config["mass"]["lower_bound_kg"]
UPPER_BOUND_MASS  = sweep_config["mass"]["upper_bound_kg"]
MASS_STEP         = sweep_config["mass"]["step_kg"]
# unitless gear ratio
LOWER_BOUND_RATIO = sweep_config["ratio"]["lower_bound"]
UPPER_BOUND_RATIO = sweep_config["ratio"]["upper_bound"]
RATIO_STEP        = sweep_config["ratio"]["step"]
# mm wheel diameter
LOWER_BOUND_DIAM  = sweep_config["diameter_mm"]["lower_bound"]
UPPER_BOUND_DIAM  = sweep_config["diameter_mm"]["upper_bound"]
DIAM_STEP         = sweep_config["diameter_mm"]["step"]

# Create 1-D arrays of values for each parameter
veh_mass_vals = np.arange(LOWER_BOUND_MASS, UPPER_BOUND_MASS + MASS_STEP, MASS_STEP)
ratio_vals    = np.arange(LOWER_BOUND_RATIO, UPPER_BOUND_RATIO + RATIO_STEP, RATIO_STEP)
diam_vals     = np.arange(LOWER_BOUND_DIAM, UPPER_BOUND_DIAM + DIAM_STEP, DIAM_STEP)


def _format_array(label, lower, upper, values, unit=None):
    gray = "\x1b[90m"
    reset = "\x1b[0m"
    prefix = f"{label} from {lower} to {upper}{f' [{unit}]' if unit else ''}"
    padded_prefix = f"{prefix:<35}{gray} "
    body = np.array2string(
        values,
        separator=", ",
        max_line_width=100,
        edgeitems=2,
        threshold=4,
        formatter={
            "float_kind": lambda x: f"{x:.6f}".rstrip("0").rstrip(".")
        },
    )
    formatted = padded_prefix + body.replace("\n", "\n" + " " * len(padded_prefix))
    return f"{formatted}{reset}"


# Compute total grid size
N_M = len(veh_mass_vals)
N_R = len(ratio_vals)
N_D = len(diam_vals)
N_TOTAL = N_SPRINGS * N_M * N_R * N_D

print()
print("=" * 57)
print(f"  Multi-objective optimizer for target speed {TARGET_SPEED_KMH} km/h")
print("=" * 57)
print()
print(f"Spring catalogue: {N_SPRINGS} entries from {os.path.basename(SPRINGS_FILE)}")
print()
print(_format_array("Mass", LOWER_BOUND_MASS, UPPER_BOUND_MASS, veh_mass_vals, "kg"))
print(_format_array("Ratio", LOWER_BOUND_RATIO, UPPER_BOUND_RATIO, ratio_vals))
print(_format_array("Diameter", LOWER_BOUND_DIAM, UPPER_BOUND_DIAM, diam_vals, "mm"))
print()
print("Drivetrain efficiency:", DRIVETRAIN_EFF)
print("Friction coefficient:", FRICTION_COEFF)
print("Safety factor:", SAFETY_FACTOR)

print()
print(f"Grid: {N_SPRINGS} springs x {N_M} vehicle masses x {N_R} ratios "
      f"x {N_D} diameters = {N_TOTAL:,} combinations")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Spring-level arrays
# ═══════════════════════════════════════════════════════════════════════════

# Create 1-D arrays of spring parameters, aligned with spring_names order
torque_arr   = np.array([springs_data[s]["max_torque_nm"]    for s in spring_names])
theta0_arr   = np.array([springs_data[s]["max_rotation_deg"] for s in spring_names])
spr_mass_arr = np.array([springs_data[s]["mass_kg"]          for s in spring_names])
k_theta_arr  = torque_arr / theta0_arr
energy_arr   = sm.stored_energy(k_theta_arr, theta0_arr)

# ═══════════════════════════════════════════════════════════════════════════
# Vectorised grid computation
# ═══════════════════════════════════════════════════════════════════════════

# Reshape for 4-D broadcasting:  (spring, veh_mass, ratio, diam)
torque_4d   = torque_arr   [:, None, None, None]
theta0_4d   = theta0_arr   [:, None, None, None]
k_th_4d     = k_theta_arr  [:, None, None, None]
E_st_4d     = energy_arr   [:, None, None, None]
spr_m_4d    = spr_mass_arr [:, None, None, None]       # spring mass [kg]
veh_m_4d    = veh_mass_vals[None, :, None, None]       # vehicle mass [kg]
rho_4d      = ratio_vals   [None, None, :, None]
R_4d        = diam_vals    [None, None, None, :] / 2000.0

# Total mass that the spring must accelerate
total_m_4d = veh_m_4d + spr_m_4d                       # (S,M,1,1)

# Energy budget — uses total mass
avail_E = DRIVETRAIN_EFF * E_st_4d                              # (S,1,1,1)
req_E   = 0.5 * total_m_4d * TARGET_SPEED_MS ** 2 * SAFETY_FACTOR  # (S,M,1,1)

# Forces — traction depends on total weight on the ground
peak_F = sm.max_force_ground(torque_4d, rho_4d, R_4d, DRIVETRAIN_EFF)
trac_L = sm.traction_limit(total_m_4d, GRAVITY, N_DRIVING_WHEELS,
                            N_WHEELS, FRICTION_COEFF, SAFETY_FACTOR)

# Hard constraints
feasible = (peak_F <= trac_L) & (avail_E >= req_E)

# Metrics — all dynamics use total mass
v_max_4d        = sm.max_speed(theta0_4d, total_m_4d, k_th_4d, DRIVETRAIN_EFF)
w_4d            = sm.omega(rho_4d, R_4d, total_m_4d, k_th_4d, DRIVETRAIN_EFF)
ratio_vt        = np.clip(TARGET_SPEED_MS / v_max_4d, 0.0, 1.0)

peak_spd_kmh_4d = v_max_4d * 3.6
t_target_4d     = np.arcsin(ratio_vt) / w_4d
x_rel_4d        = sm.release_distance(rho_4d, R_4d, theta0_4d)
d_target_4d     = x_rel_4d * (1.0 - np.sqrt(np.maximum(1.0 - ratio_vt ** 2, 0.0)))
margin_N_4d     = trac_L - peak_F
margin_pct_4d   = 100.0 * margin_N_4d / trac_L
accel_4d        = peak_F / total_m_4d

# ═══════════════════════════════════════════════════════════════════════════
# Flatten and filter to feasible
# ═══════════════════════════════════════════════════════════════════════════

# Create 1-D arrays for each parameter/metric, aligned by raveled index.  Then
# we can apply the same boolean mask to all of them to get the feasible subset.  The
# spring index is repeated so we can identify which spring each row corresponds to.
full_shape = (N_SPRINGS, N_M, N_R, N_D)

s_flat     = np.broadcast_to(np.arange(N_SPRINGS)[:, None, None, None], full_shape).ravel()
vm_flat    = np.broadcast_to(veh_m_4d,   full_shape).ravel()    # vehicle mass
sm_flat    = np.broadcast_to(spr_m_4d,   full_shape).ravel()    # spring mass
tm_flat    = np.broadcast_to(total_m_4d, full_shape).ravel()    # total mass
rho_flat   = np.broadcast_to(rho_4d,     full_shape).ravel()
diam_flat  = np.broadcast_to(diam_vals[None, None, None, :], full_shape).ravel()

spd_flat = np.broadcast_to(peak_spd_kmh_4d, full_shape).ravel()
tt_flat  = np.broadcast_to(t_target_4d,     full_shape).ravel()
dt_flat  = np.broadcast_to(d_target_4d,     full_shape).ravel()
mn_flat  = np.broadcast_to(margin_N_4d,     full_shape).ravel()
mp_flat  = np.broadcast_to(margin_pct_4d,   full_shape).ravel()
ac_flat  = np.broadcast_to(accel_4d,        full_shape).ravel()

# Apply feasibility mask to filter all arrays to the feasible subset
mask   = feasible.ravel()
n_feas = int(mask.sum())
print(f"Filtered to total feasible: {n_feas:,} / {N_TOTAL:,} ({100.0 * n_feas / N_TOTAL:.1f}%)")

if n_feas == 0:
    print("No feasible combinations found. Exiting.")
    raise SystemExit(1)

# Create 1-D arrays for the feasible subset (same order across all arrays)
F_s    = s_flat   [mask].copy()
F_vm   = vm_flat  [mask].copy()
F_sm   = sm_flat  [mask].copy()
F_tm   = tm_flat  [mask].copy()
F_rho  = rho_flat [mask].copy()
F_diam = diam_flat[mask].copy()
F_spd  = spd_flat [mask].copy()
F_tt   = tt_flat  [mask].copy()
F_dt   = dt_flat  [mask].copy()
F_mn   = mn_flat  [mask].copy()
F_mp   = mp_flat  [mask].copy()
F_ac   = ac_flat  [mask].copy()

# ═══════════════════════════════════════════════════════════════════════════
# Save results.csv
# ═══════════════════════════════════════════════════════════════════════════

# Save the full feasible grid to results.csv, sorted by peak speed.
sort_idx = np.argsort(-F_spd)

if args.save_all:
    print(f"Saving all_feasible.csv ({n_feas:,} rows) ...")
    with open(os.path.join(RESULTS_DIR, "all_feasible.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["spring", "vehicle_mass_kg", "spring_mass_kg",
                         "total_mass_kg", "gear_ratio", "wheel_diam_mm",
                         "peak_speed_kmh", "time_to_target_s", "dist_to_target_m",
                         "traction_margin_n", "traction_margin_pct",
                         "initial_accel_m_s2"])
        for i in sort_idx:
            writer.writerow([
                spring_names[int(F_s[i])],
                f"{F_vm[i]:.2f}", f"{F_sm[i]:.4f}", f"{F_tm[i]:.4f}",
                f"{F_rho[i]:.0f}",  f"{F_diam[i]:.0f}",
                f"{F_spd[i]:.2f}", f"{F_tt[i]:.4f}",   f"{F_dt[i]:.4f}",
                f"{F_mn[i]:.4f}",  f"{F_mp[i]:.2f}",   f"{F_ac[i]:.4f}",
            ])
else:
    print(f"Skipping all_feasible.csv (pass --save-all to write it)")

# ═══════════════════════════════════════════════════════════════════════════
# Pareto front extraction
# ═══════════════════════════════════════════════════════════════════════════

def pareto_front_indices(costs, desc="Pareto extraction"):
    """
    Return indices of Pareto-optimal (non-dominated) rows.

    Parameters
    ----------
    costs : ndarray, shape (n, k)
        All objectives to **minimise**.

    Returns
    -------
    ndarray of int — indices into *costs* that form the Pareto front.
    """
    n, k = costs.shape

    # Process likely-good points first (low sum of normalised objectives)
    mn  = costs.min(axis=0)
    rng = np.maximum(costs.max(axis=0) - mn, 1e-12)
    order = np.argsort(((costs - mn) / rng).sum(axis=1))

    p_ids   = np.empty(n, dtype=int)
    p_costs = np.empty((n, k))
    n_p     = 0

    for raw in tqdm(order, desc=desc, unit="pt", ncols=110):
        c = costs[raw]
        if n_p > 0:
            pc = p_costs[:n_p]
            # Is c dominated by any current Pareto member?
            if np.any(np.all(pc <= c, axis=1) & np.any(pc < c, axis=1)):
                continue
            # Drop members dominated by c
            keep = ~(np.all(c <= pc, axis=1) & np.any(c < pc, axis=1))
            if not np.all(keep):
                kw    = np.where(keep)[0]
                new_n = len(kw)
                p_ids  [:new_n] = p_ids  [kw]
                p_costs[:new_n] = p_costs[kw]
                n_p = new_n

        p_ids  [n_p] = raw
        p_costs[n_p] = c
        n_p += 1

    return p_ids[:n_p].copy()

print()
print("Extracting Pareto front from grid ...")
# Three objectives: maximize speed (negate), minimize total mass, minimize gear ratio
grid_costs      = np.column_stack([-F_spd, F_tm, F_rho])
grid_pareto_idx = pareto_front_indices(grid_costs, desc="Pareto (grid)")
print(f"Found {len(grid_pareto_idx):,} Pareto-optimal points in grid.")

# ═══════════════════════════════════════════════════════════════════════════
# Scipy per-spring continuous refinement
# ═══════════════════════════════════════════════════════════════════════════

print()
print("Refining with scipy optimiser ...\x1b[90m")

OPT_BOUNDS = [(LOWER_BOUND_MASS, UPPER_BOUND_MASS),                   # vehicle mass [kg]
              (LOWER_BOUND_RATIO, UPPER_BOUND_RATIO),                 # gear ratio
              (LOWER_BOUND_DIAM / 2000.0, UPPER_BOUND_DIAM / 2000.0)] # wheel radius [m]

opt_rows = []

for si in range(N_SPRINGS):
    sname = spring_names[si]
    tau   = torque_arr[si]
    th0   = theta0_arr[si]
    kth   = k_theta_arr[si]
    E_spr = energy_arr[si]
    m_spr = spr_mass_arr[si]

    # Quick check: enough energy at lowest total mass?
    min_total = veh_mass_vals[0] + m_spr
    if DRIVETRAIN_EFF * E_spr < 0.5 * min_total * TARGET_SPEED_MS ** 2 * SAFETY_FACTOR:
        print(f"  [{si+1:2d}/{N_SPRINGS}] {sname} ({m_spr*1000:.0f} g)"
              f" - insufficient energy")
        continue

    # Constraint helpers — x = [vehicle_mass, rho, R]
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

    cons = [NonlinearConstraint(_noslip, 0, np.inf),
            NonlinearConstraint(_energy, 0, np.inf)]

    def _obj_speed(x, _th0=th0, _kth=kth, _ms=m_spr):
        mt = x[0] + _ms
        return float(-sm.max_speed(_th0, mt, _kth, DRIVETRAIN_EFF) * 3.6)

    def _obj_mass(x, _ms=m_spr):
        return float(x[0] + _ms)  # minimize total mass

    def _obj_ratio(x):
        return float(x[1])  # minimize gear ratio

    n_found = 0
    for label, obj_fn in [("speed", _obj_speed),
                           ("mass",  _obj_mass),
                           ("ratio", _obj_ratio)]:
        try:
            res = differential_evolution(
                obj_fn, OPT_BOUNDS, constraints=cons,
                seed=42, maxiter=200, tol=1e-8, polish=True)

            if not res.success:
                continue
            mv, rho_, R_ = res.x
            mt = mv + m_spr
            if _noslip(res.x) < -1e-6 or _energy(res.x) < -1e-6:
                continue

            tl = float(sm.traction_limit(mt, GRAVITY, N_DRIVING_WHEELS,
                                          N_WHEELS, FRICTION_COEFF,
                                          SAFETY_FACTOR))
            pf = float(sm.max_force_ground(tau, rho_, R_, DRIVETRAIN_EFF))
            mn = tl - pf
            opt_rows.append({
                "s":    si,
                "vm":   mv,
                "sm":   m_spr,
                "tm":   mt,
                "rho":  rho_,
                "diam": R_ * 2000.0,
                "spd":  float(sm.max_speed(th0, mt, kth, DRIVETRAIN_EFF) * 3.6),
                "tt":   float(sm.time_to_speed(TARGET_SPEED_MS, rho_, R_,
                                                th0, mt, kth, DRIVETRAIN_EFF)),
                "dt":   float(sm.distance_to_speed(TARGET_SPEED_MS, rho_, R_,
                                                    th0, mt, kth,
                                                    DRIVETRAIN_EFF)),
                "mn":   mn,
                "mp":   100.0 * mn / tl,
                "ac":   pf / mt,
            })
            n_found += 1
        except Exception:
            pass

    print(f"  [{si+1:2d}/{N_SPRINGS}] {sname} ({m_spr*1000:.0f} g)"
          f" - {n_found} optima found")

print(f"\x1b[0mOptimiser added {len(opt_rows)} new points across all springs.")

# ═══════════════════════════════════════════════════════════════════════════
# Merge grid + optimiser results and re-extract Pareto
# ═══════════════════════════════════════════════════════════════════════════

if opt_rows:
    o = {k: np.array([r[k] for r in opt_rows]) for k in
         ["s", "vm", "sm", "tm", "rho", "diam",
          "spd", "tt", "dt", "mn", "mp", "ac"]}

    A_s    = np.concatenate([F_s,    o["s"]])
    A_vm   = np.concatenate([F_vm,   o["vm"]])
    A_sm   = np.concatenate([F_sm,   o["sm"]])
    A_tm   = np.concatenate([F_tm,   o["tm"]])
    A_rho  = np.concatenate([F_rho,  o["rho"]])
    A_diam = np.concatenate([F_diam, o["diam"]])
    A_spd  = np.concatenate([F_spd,  o["spd"]])
    A_tt   = np.concatenate([F_tt,   o["tt"]])
    A_dt   = np.concatenate([F_dt,   o["dt"]])
    A_mn   = np.concatenate([F_mn,   o["mn"]])
    A_mp   = np.concatenate([F_mp,   o["mp"]])
    A_ac   = np.concatenate([F_ac,   o["ac"]])
else:
    A_s, A_vm, A_sm, A_tm  = F_s, F_vm, F_sm, F_tm
    A_rho, A_diam           = F_rho, F_diam
    A_spd, A_tt, A_dt       = F_spd, F_tt, F_dt
    A_mn, A_mp, A_ac        = F_mn, F_mp, F_ac

print()
print("Re-extracting combined Pareto front ...")
# Three objectives: maximize speed (negate), minimize total mass, minimize gear ratio
all_costs  = np.column_stack([-A_spd, A_tm, A_rho])
pareto_idx = pareto_front_indices(all_costs, desc="Pareto (combined)")
n_pareto   = len(pareto_idx)
print(f"Final Pareto-optimal points: {n_pareto:,}")

# Boolean mask over the combined (A_*) arrays
is_pareto = np.zeros(len(A_s), dtype=bool)
is_pareto[pareto_idx] = True

P_s    = A_s   [pareto_idx];  P_vm   = A_vm  [pareto_idx]
P_sm   = A_sm  [pareto_idx];  P_tm   = A_tm  [pareto_idx]
P_rho  = A_rho [pareto_idx];  P_diam = A_diam[pareto_idx]
P_spd  = A_spd [pareto_idx];  P_tt   = A_tt  [pareto_idx]
P_dt   = A_dt  [pareto_idx];  P_mn   = A_mn  [pareto_idx]
P_mp   = A_mp  [pareto_idx];  P_ac   = A_ac  [pareto_idx]

# ═══════════════════════════════════════════════════════════════════════════
# Save pareto.csv
# ═══════════════════════════════════════════════════════════════════════════

print(f"Saving pareto_optimal.csv ({n_pareto:,} rows) ...")
p_sort = np.argsort(-P_spd)

with open(os.path.join(RESULTS_DIR, "pareto_optimal.csv"), "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["spring", "vehicle_mass_kg", "spring_mass_kg",
                     "total_mass_kg", "gear_ratio", "wheel_diam_mm",
                     "peak_speed_kmh", "time_to_target_s", "dist_to_target_m",
                     "traction_margin_n", "traction_margin_pct",
                     "initial_accel_m_s2"])
    for i in p_sort:
        writer.writerow([
            spring_names[int(P_s[i])],
            f"{P_vm[i]:.2f}", f"{P_sm[i]:.4f}", f"{P_tm[i]:.4f}",
            f"{P_rho[i]:.1f}",  f"{P_diam[i]:.1f}",
            f"{P_spd[i]:.2f}", f"{P_tt[i]:.4f}",   f"{P_dt[i]:.4f}",
            f"{P_mn[i]:.4f}",  f"{P_mp[i]:.2f}",   f"{P_ac[i]:.4f}",
        ])

# ═══════════════════════════════════════════════════════════════════════════
# Key configurations + Top-5 rankings  (printed AND saved to txt)
# ═══════════════════════════════════════════════════════════════════════════

def _cfg_block(tag, idx):
    """Format a single configuration block."""
    s = spring_names[int(P_s[idx])].upper()
    return (
        f"  {tag}\n"
        f"    Spring   : {s} ({P_sm[idx]*1000:.0f} g)\n"
        f"    Vehicle  : {P_vm[idx]:.2f} kg  |  Total: {P_tm[idx]:.2f} kg\n"
        f"    Gearing  : ratio {P_rho[idx]:.1f}  |  wheel D {P_diam[idx]:.0f} mm\n"
        f"    Speed    : {P_spd[idx]:.2f} km/h\n"
        f"    Target   : reached in {P_tt[idx]:.2f} s / {P_dt[idx]:.2f} m\n"
        f"    Traction : {P_mn[idx]:.2f} N margin ({P_mp[idx]:.1f}%)\n"
    )

def _table_header():
    return (
        f"  {'#':>2s}  {'Spring':>8s}  {'Spr(g)':>6s}  {'Veh(kg)':>7s}  "
        f"{'Tot(kg)':>7s}  {'Ratio':>5s}  {'D(mm)':>5s}  "
        f"{'Speed':>7s}  {'Time':>6s}  {'Dist':>6s}\n"
        + "  " + "-" * 82
    )

def _table_row(rank, idx):
    return (
        f"  {rank:>2d}  {spring_names[int(P_s[idx])].upper():>8s}  "
        f"{P_sm[idx]*1000:>6.0f}  {P_vm[idx]:>7.2f}  {P_tm[idx]:>7.2f}  "
        f"{P_rho[idx]:>5.1f}  {P_diam[idx]:>5.0f}  "
        f"{P_spd[idx]:>6.2f}  {P_tt[idx]:>6.2f}  {P_dt[idx]:>6.2f}"
    )

# --- Identify key configurations ---

i_fastest  = np.argmax(P_spd)
i_lightest = np.argmin(P_tm)
i_simplest = np.argmin(P_rho)

# Balanced: normalise the 3 objectives to [0,1] and pick the point that
# maximises (norm_speed - norm_mass - norm_ratio), i.e. equal weight to each.
p_spd_n = (P_spd - P_spd.min()) / max(P_spd.max() - P_spd.min(), 1e-12)
p_tm_n  = (P_tm  - P_tm.min())  / max(P_tm.max()  - P_tm.min(),  1e-12)
p_rho_n = (P_rho - P_rho.min()) / max(P_rho.max() - P_rho.min(), 1e-12)
balanced_score = p_spd_n - p_tm_n - p_rho_n
i_balanced = np.argmax(balanced_score)

# --- Build text ---

lines = []
lines.append("=" * 68)
lines.append("  KEY CONFIGURATIONS  (from Pareto front)")
lines.append("=" * 68)
lines.append("")
lines.append(_cfg_block("FASTEST", i_fastest))
lines.append(_cfg_block("LIGHTEST", i_lightest))
lines.append(_cfg_block("SIMPLEST GEARING", i_simplest))
lines.append(_cfg_block(
    f"BALANCED  (score {balanced_score[i_balanced]:.3f})", i_balanced))

lines.append("=" * 68)
lines.append("  TOP 5 RANKINGS  (Pareto-optimal only)")
lines.append("=" * 68)

for title, indices in [
    ("Ranked by Peak Speed (descending)",  np.argsort(-P_spd)[:5]),
    ("Ranked by Total Mass (ascending)",   np.argsort(P_tm) [:5]),
    ("Ranked by Gear Ratio (ascending)",   np.argsort(P_rho)[:5]),
    ("Ranked by Balanced Score (descending)", np.argsort(-balanced_score)[:5]),
]:
    lines.append(f"\n  {title}")
    lines.append(_table_header())
    for rank, i in enumerate(indices, 1):
        lines.append(_table_row(rank, i))

lines.append("")

report = "\n".join(lines)
print("\n" + report)

with open(os.path.join(RESULTS_DIR, "key_configurations.txt"), "w") as fh:
    fh.write(report + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

print("Generating plots ...")

# -- Shared colour setup --
# Remap colors to only span springs that appear on the Pareto front.
pareto_spring_ids = np.unique(P_s.astype(int))
n_pareto_springs  = len(pareto_spring_ids)

# Create a lookup: original spring index -> color index [0, n_pareto_springs-1]
spring_to_color = {sid: i for i, sid in enumerate(pareto_spring_ids)}

def _map_spring_to_color(spring_indices):
    """Map spring IDs to color indices [0, n_pareto_springs-1]."""
    return np.array([spring_to_color.get(int(s), 0) for s in spring_indices])

cmap = plt.cm.turbo
norm = plt.Normalize(0, n_pareto_springs - 1)

def _add_spring_cbar(fig, ax_or_axes, *, pad=0.08):
    sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax_or_axes, shrink=0.55, pad=pad,
                        aspect=30)
    cbar.set_ticks(np.arange(n_pareto_springs))
    cbar.set_ticklabels([spring_names[sid].upper() for sid in pareto_spring_ids])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Spring", fontsize=9)
    return cbar

# -- Subsample background feasible cloud --
MAX_BG   = 8000
rng_plot = np.random.default_rng(42)
n_all    = len(A_s)
bg_idx   = (rng_plot.choice(n_all, MAX_BG, replace=False)
            if n_all > MAX_BG else np.arange(n_all))

BG_s   = A_s[bg_idx];    BG_spd  = A_spd[bg_idx]
BG_tm  = A_tm[bg_idx];   BG_rho  = A_rho[bg_idx]
BG_tt  = A_tt[bg_idx];   BG_dt   = A_dt[bg_idx]
BG_vm  = A_vm[bg_idx];   BG_sm   = A_sm[bg_idx]
BG_diam = A_diam[bg_idx]

# ---- 1. 3D Pareto surface ------------------------------------------------

fig = plt.figure(figsize=(13, 9))
ax  = fig.add_subplot(111, projection="3d")
ax.view_init(elev=25, azim=135)

ax.scatter(BG_spd, BG_tm, BG_rho,
           c=_map_spring_to_color(BG_s), cmap=cmap, norm=norm,
           s=2, alpha=0.06, edgecolors="none", rasterized=True)
ax.scatter(P_spd, P_tm, P_rho,
           c=_map_spring_to_color(P_s), cmap=cmap, norm=norm,
           s=30, alpha=0.9, edgecolors="k", linewidths=0.35)

ax.set_xlabel("Peak Speed [km/h]",  fontsize=10, labelpad=10)
ax.set_ylabel("Total Mass [kg]",    fontsize=10, labelpad=10)
ax.set_zlabel("Gear Ratio",         fontsize=10, labelpad=8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.15)
ax.set_title("Speed / Mass / Gear Ratio: Pareto front (bold) over feasible manifold (faint)", fontsize=12, fontweight="bold", pad=18)
_add_spring_cbar(fig, ax, pad=0.12)

# Annotation at bottom
annotation = (f"Each point = feasible configuration (spring + vehicle mass + gearing). "
              f"Bold markers = Pareto-optimal (no other config beats it on all 3 objectives). "
              f"{n_feas:,} feasible → {n_pareto} Pareto.")
fig.text(0.5, 0.01, annotation, ha="center", fontsize=7, 
         style="italic", color="0.4", wrap=True)

fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.06)
plt.savefig(os.path.join(GRAPHS_DIR, "pareto_3d_speed_mass_ratio.png"), dpi=DPI)
if args.interactive:
    plt.show()
plt.close(fig)
print("  graphs/pareto_3d_speed_mass_ratio.png")

# ---- 2. 2D projections ---------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

specs = [
    ("Peak Speed [km/h]", "Total Mass [kg]",  BG_spd, BG_tm,  P_spd, P_tm),
    ("Peak Speed [km/h]", "Gear Ratio",        BG_spd, BG_rho, P_spd, P_rho),
    ("Total Mass [kg]",   "Gear Ratio",        BG_tm,  BG_rho, P_tm,  P_rho),
]

for ax, (xl, yl, bx, by, fx, fy) in zip(axes, specs):
    ax.scatter(bx, by, c=_map_spring_to_color(BG_s), cmap=cmap, norm=norm,
               s=3, alpha=0.10, edgecolors="none", rasterized=True)
    ax.scatter(fx, fy, c=_map_spring_to_color(P_s), cmap=cmap, norm=norm,
               s=28, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_box_aspect(1)

fig.suptitle("2D Projections: Pareto front (bold) over feasible manifold (faint)", fontsize=12, fontweight="bold")
_add_spring_cbar(fig, axes, pad=0.01)

# Annotation at bottom
annotation = (f"Three 2D slices of the 3-objective Pareto front. "
              f"Each projection shows the trade-off between two objectives. "
              f"Colors indicate which spring is used. "
              f"Bold markers = Pareto-optimal")
fig.text(0.5, -0.02, annotation, ha="center", fontsize=7, 
         style="italic", color="0.4", wrap=True, transform=fig.transFigure)
plt.savefig(os.path.join(GRAPHS_DIR, "pareto_2d_projections.png"), dpi=DPI, bbox_inches='tight')
if args.interactive:
    plt.show()
plt.close(fig)
print("  graphs/pareto_2d_projections.png")

# ---- 3. Parallel coordinates ---------------------------------------------

fig, ax = plt.subplots(figsize=(15, 6.5))
fig.subplots_adjust(left=0.04, right=0.86, top=0.88, bottom=0.15)

pc_labels = ["Veh mass\n(kg)", "Spring\nmass (g)", "Total\nmass (kg)",
             "Gear\nratio", "Wheel D\n(mm)",
             "Speed\n(km/h)", "Time\n(s)", "Dist\n(m)"]

pc_fg = np.column_stack([P_vm,          P_sm * 1000,          P_tm,
                         P_rho,          P_diam,
                         P_spd,          P_tt,                 P_dt])
pc_bg = np.column_stack([BG_vm,          BG_sm * 1000,        BG_tm,
                         BG_rho,          BG_diam,
                         BG_spd,          BG_tt,               BG_dt])

pc_all  = np.vstack([pc_fg, pc_bg])
pc_min  = pc_all.min(axis=0);  pc_max = pc_all.max(axis=0)
pc_rng  = np.maximum(pc_max - pc_min, 1e-12)
pc_bg_n = (pc_bg - pc_min) / pc_rng
pc_fg_n = (pc_fg - pc_min) / pc_rng

xs        = np.arange(len(pc_labels))
bg_colors = cmap(norm(_map_spring_to_color(BG_s)))
fg_colors = cmap(norm(_map_spring_to_color(P_s)))

max_bg_lines = 2000
draw_bg = (rng_plot.choice(len(pc_bg_n), max_bg_lines, replace=False)
           if len(pc_bg_n) > max_bg_lines else np.arange(len(pc_bg_n)))

for i in draw_bg:
    ax.plot(xs, pc_bg_n[i], color=bg_colors[i], alpha=0.05, linewidth=0.35)

for i in range(len(pc_fg_n)):
    ax.plot(xs, pc_fg_n[i], color=fg_colors[i], alpha=0.65, linewidth=1.5)

for j in range(len(xs)):
    ax.axvline(j, color="k", linewidth=0.5, alpha=0.25)
    ax.annotate(f"{pc_min[j]:.1f}", xy=(j, -0.04), ha="center", va="top",
                fontsize=7, color="0.35")
    ax.annotate(f"{pc_max[j]:.1f}", xy=(j,  1.04), ha="center", va="bottom",
                fontsize=7, color="0.35")

ax.set_xticks(xs)
ax.set_xticklabels(pc_labels, fontsize=9)
ax.set_ylim(-0.10, 1.12)
ax.set_yticks([])
ax.set_title("Parallel Coordinates: Pareto front (bold) over feasible manifold (faint)", fontsize=11, fontweight="bold", pad=12)
_add_spring_cbar(fig, ax, pad=0.03)

# Annotation at bottom
annotation = (f"Each line = one configuration; vertical axes show normalized values. "
              f"Bold lines are Pareto-optimal. Trace horizontally to compare metrics across one configuration.")
fig.text(0.5, 0.03, annotation, ha="center", fontsize=7, 
         style="italic", color="0.4", wrap=True)

fig.subplots_adjust(bottom=0.12)
plt.savefig(os.path.join(GRAPHS_DIR, "pareto_parallel_coordinates.png"), dpi=DPI)
if args.interactive:
    plt.show()
plt.close(fig)
print("  graphs/pareto_parallel_coordinates.png")

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════

print(f"""
Output files:
  results/all_feasible.csv             {'({n_feas:,} rows)' if args.save_all else '(skipped)'}
  results/pareto_optimal.csv           ({n_pareto:,} rows)
  results/key_configurations.txt
  graphs/pareto_3d_speed_mass_ratio.png
  graphs/pareto_2d_projections.png
  graphs/pareto_parallel_coordinates.png

Done.
""")
