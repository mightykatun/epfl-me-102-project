#!/usr/bin/env python3
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
    python explore.py
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)
from scipy.optimize import differential_evolution, NonlinearConstraint

import spring_model as sm

# ═══════════════════════════════════════════════════════════════════════════
# Constants  (same values as spring.py)
# ═══════════════════════════════════════════════════════════════════════════

TARGET_SPEED_KMH = 15.0
TARGET_SPEED_MS  = TARGET_SPEED_KMH / 3.6
FRICTION_COEFF   = 0.7
SAFETY_FACTOR    = 1.5
DRIVETRAIN_EFF   = 0.92
N_WHEELS         = 3
N_DRIVING_WHEELS = 2
GRAVITY          = 9.81
DPI              = 300

# ═══════════════════════════════════════════════════════════════════════════
# Load spring catalogue
# ═══════════════════════════════════════════════════════════════════════════

SPRINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "springs.txt")
springs_data = sm.load_springs(SPRINGS_FILE)
spring_names = sorted(springs_data.keys())
N_SPRINGS    = len(spring_names)

# ═══════════════════════════════════════════════════════════════════════════
# Sweep ranges
# ═══════════════════════════════════════════════════════════════════════════

# kg not counting spring mass
LOWER_BOUND_MASS  = 1.50
UPPER_BOUND_MASS  = 4
MASS_STEP         = 0.125
# unitless gear ratio
LOWER_BOUND_RATIO = 10.0
UPPER_BOUND_RATIO = 80.0
RATIO_STEP        = 2.5
# mm wheel diameter
LOWER_BOUND_DIAM  = 50.0
UPPER_BOUND_DIAM  = 200.0
DIAM_STEP         = 5.0

# Create 1-D arrays of values for each parameter
veh_mass_vals = np.arange(LOWER_BOUND_MASS, UPPER_BOUND_MASS + MASS_STEP, MASS_STEP)
ratio_vals    = np.arange(LOWER_BOUND_RATIO, UPPER_BOUND_RATIO + RATIO_STEP, RATIO_STEP)
diam_vals     = np.arange(LOWER_BOUND_DIAM, UPPER_BOUND_DIAM + DIAM_STEP, DIAM_STEP)

# Compute total grid size
N_M = len(veh_mass_vals)
N_R = len(ratio_vals)
N_D = len(diam_vals)
N_TOTAL = N_SPRINGS * N_M * N_R * N_D

print(f"Grid: {N_SPRINGS} springs x {N_M} vehicle masses x {N_R} ratios "
      f"x {N_D} diameters = {N_TOTAL:,} combinations")

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

print("Computing grid metrics ...")

# Reshape for 4-D broadcasting:  (spring, veh_mass, ratio, diam)
torque_4d   = torque_arr  [:, None, None, None]
theta0_4d   = theta0_arr  [:, None, None, None]
k_th_4d     = k_theta_arr [:, None, None, None]
E_st_4d     = energy_arr  [:, None, None, None]
spr_m_4d    = spr_mass_arr[:, None, None, None]       # spring mass [kg]
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

print("Filtering to feasible combinations ...")

# Create 1-D arrays for each parameter/metric, aligned by raveled index.  Then
# we can apply the same boolean mask to all of them to get the feasible subset.  The
# spring index is repeated so we can identify which spring each row corresponds to.
full_shape = (N_SPRINGS, N_M, N_R, N_D)

s_flat     = np.broadcast_to(np.arange(N_SPRINGS)[:, None, None, None], full_shape).ravel()
vm_flat    = np.broadcast_to(veh_m_4d,  full_shape).ravel()       # vehicle mass
sm_flat    = np.broadcast_to(spr_m_4d,  full_shape).ravel()       # spring mass
tm_flat    = np.broadcast_to(total_m_4d, full_shape).ravel()      # total mass
rho_flat   = np.broadcast_to(rho_4d,    full_shape).ravel()
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
print(f"Feasible: {n_feas:,} / {N_TOTAL:,} ({100.0 * n_feas / N_TOTAL:.1f}%)")

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
print(f"Saving results.csv ({n_feas:,} rows) ...")
sort_idx = np.argsort(-F_spd)

with open("results.csv", "w", newline="") as fh:
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

# ═══════════════════════════════════════════════════════════════════════════
# Pareto front extraction
# ═══════════════════════════════════════════════════════════════════════════

def pareto_front_indices(costs):
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

    for raw in order:
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


print("Extracting Pareto front from grid ...")
# Three objectives: maximize speed (negate), minimize total mass, minimize gear ratio
grid_costs      = np.column_stack([-F_spd, F_tm, F_rho])
grid_pareto_idx = pareto_front_indices(grid_costs)
print(f"Grid Pareto-optimal: {len(grid_pareto_idx):,}")

# ═══════════════════════════════════════════════════════════════════════════
# Scipy per-spring continuous refinement
# ═══════════════════════════════════════════════════════════════════════════

print("Refining with scipy optimiser ...")

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
              f" — skipped (insufficient energy)")
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
          f" — {n_found} optima found")

print(f"Optimiser points added: {len(opt_rows)}")

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

print("Re-extracting combined Pareto front ...")
# Three objectives: maximize speed (negate), minimize total mass, minimize gear ratio
all_costs  = np.column_stack([-A_spd, A_tm, A_rho])
pareto_idx = pareto_front_indices(all_costs)
n_pareto   = len(pareto_idx)
print(f"Final Pareto-optimal: {n_pareto:,}")

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

print(f"Saving pareto.csv ({n_pareto:,} rows) ...")
p_sort = np.argsort(-P_spd)

with open("pareto.csv", "w", newline="") as fh:
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
# Print top-5 summary
# ═══════════════════════════════════════════════════════════════════════════

def _row_str(idx):
    return (f"  {spring_names[int(P_s[idx])]:>8s} "
            f"({P_sm[idx]*1000:.0f} g)  "
            f"veh={P_vm[idx]:.2f} tot={P_tm[idx]:.2f} kg  "
            f"rho={P_rho[idx]:.1f}  D={P_diam[idx]:.0f} mm  |  "
            f"spd={P_spd[idx]:.2f} km/h  "
            f"t={P_tt[idx]:.2f} s  d={P_dt[idx]:.2f} m")

print("\n=== Top 5 by Peak Speed ===")
for i in np.argsort(-P_spd)[:5]:
    print(_row_str(i))

print("\n=== Top 5 by Minimum Total Mass ===")
for i in np.argsort(P_tm)[:5]:
    print(_row_str(i))

print("\n=== Top 5 by Minimum Gear Ratio ===")
for i in np.argsort(P_rho)[:5]:
    print(_row_str(i))

# ═══════════════════════════════════════════════════════════════════════════
# Plots — show full feasible manifold + Pareto front highlighted
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating plots ...")

cmap = plt.cm.turbo
norm = plt.Normalize(0, N_SPRINGS - 1)

def _add_spring_cbar(fig, ax_or_axes):
    sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_cb.set_array([])
    cbar = fig.colorbar(sm_cb, ax=ax_or_axes, shrink=0.6, pad=0.02)
    tick_pos = np.arange(0, N_SPRINGS, max(1, N_SPRINGS // 10))
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([spring_names[i].upper() for i in tick_pos])
    cbar.set_label("Spring")
    return cbar

# Subsample the full manifold for plotting if very large
MAX_BG = 8000
rng_plot = np.random.default_rng(42)
n_all    = len(A_s)
if n_all > MAX_BG:
    bg_idx = rng_plot.choice(n_all, MAX_BG, replace=False)
else:
    bg_idx = np.arange(n_all)

BG_s   = A_s[bg_idx];   BG_spd = A_spd[bg_idx]
BG_tt  = A_tt[bg_idx];  BG_dt  = A_dt[bg_idx]


# ---------- 1. 3D Pareto front ----------

BG_tm = A_tm[bg_idx]
BG_rho = A_rho[bg_idx]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# background: all feasible (faint)
ax.scatter(BG_spd, BG_tm, BG_rho,
           c=BG_s.astype(int), cmap=cmap, norm=norm,
           s=3, alpha=0.08, edgecolors="none")
# Pareto front (bold)
ax.scatter(P_spd, P_tm, P_rho,
           c=P_s.astype(int), cmap=cmap, norm=norm,
           s=25, alpha=0.9, edgecolors="k", linewidths=0.4)

ax.set_xlabel("Peak Speed [km/h]")
ax.set_ylabel("Total Mass [kg]")
ax.set_zlabel("Gear Ratio")
ax.set_title("Pareto Front: Speed / Mass / Gear Ratio", fontweight="bold")
_add_spring_cbar(fig, ax)
plt.tight_layout()
plt.savefig("pareto_3d.png", dpi=DPI)
plt.close(fig)
print("  pareto_3d.png")


# ---------- 2. 2D projections of the 3 objectives ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

pairs = [("Peak Speed [km/h]", "Total Mass [kg]"),
         ("Peak Speed [km/h]", "Gear Ratio"),
         ("Total Mass [kg]",   "Gear Ratio")]

bg_arrs = [(BG_spd, BG_tm), (BG_spd, BG_rho), (BG_tm, BG_rho)]
fg_arrs = [(P_spd,  P_tm),  (P_spd,  P_rho),  (P_tm,  P_rho)]

for ax, (xl, yl), (bx, by), (fx, fy) in zip(axes, pairs, bg_arrs, fg_arrs):
    ax.scatter(bx, by, c=BG_s.astype(int), cmap=cmap, norm=norm,
               s=3, alpha=0.10, edgecolors="none")
    ax.scatter(fx, fy, c=P_s.astype(int), cmap=cmap, norm=norm,
               s=25, alpha=0.85, edgecolors="k", linewidths=0.5)
    ax.set_xlabel(xl);  ax.set_ylabel(yl)
    ax.grid(True, alpha=0.3)

fig.suptitle("Pareto front — 2D projections of objectives", fontsize=13)
_add_spring_cbar(fig, axes.tolist())
plt.tight_layout()
plt.savefig("pareto_2d_panels.png", dpi=DPI)
plt.close(fig)
print("  pareto_2d_panels.png")


# ---------- 3. Top-20 tables ----------

fig, axes = plt.subplots(3, 1, figsize=(16, 16))

col_labels = ["#", "Spring", "Spr\n(g)", "Veh\n(kg)", "Tot\n(kg)",
              "Ratio", "D (mm)",
              "Speed\n(km/h)", "Time\n(s)", "Dist\n(m)",
              "Margin\n(N)", "Accel\n(m/s\u00b2)"]

table_specs = [
    ("Top 20 Pareto — ranked by Peak Speed (descending)", np.argsort(-P_spd)[:20]),
    ("Top 20 Pareto — ranked by Total Mass (ascending)",   np.argsort(P_tm) [:20]),
    ("Top 20 Pareto — ranked by Gear Ratio (ascending)",   np.argsort(P_rho)[:20]),
]

for ax, (title, indices) in zip(axes, table_specs):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    rows = []
    for rank, i in enumerate(indices, 1):
        rows.append([
            str(rank),
            spring_names[int(P_s[i])].upper(),
            f"{P_sm[i]*1000:.0f}",
            f"{P_vm[i]:.2f}", f"{P_tm[i]:.2f}",
            f"{P_rho[i]:.0f}", f"{P_diam[i]:.0f}",
            f"{P_spd[i]:.2f}", f"{P_tt[i]:.2f}", f"{P_dt[i]:.2f}",
            f"{P_mn[i]:.2f}", f"{P_ac[i]:.2f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.3)

plt.tight_layout()
plt.savefig("top20_table.png", dpi=DPI)
plt.close(fig)
print("  top20_table.png")


# ---------- 4. Parallel coordinates ----------

fig, ax = plt.subplots(figsize=(14, 6))

pc_labels = ["Veh mass\n(kg)", "Spring\nmass (g)", "Total\nmass (kg)",
             "Gear ratio", "Wheel D\n(mm)",
             "Speed\n(km/h)", "Time\n(s)", "Dist\n(m)"]

# Foreground: Pareto points.  Background: subsample of all feasible.
pc_fg = np.column_stack([P_vm, P_sm * 1000, P_tm, P_rho, P_diam,
                         P_spd, P_tt, P_dt])
pc_bg = np.column_stack([A_vm[bg_idx], A_sm[bg_idx] * 1000,
                         A_tm[bg_idx], A_rho[bg_idx], A_diam[bg_idx],
                         A_spd[bg_idx], A_tt[bg_idx], A_dt[bg_idx]])

# Normalise using the full feasible range
pc_all = np.vstack([pc_fg, pc_bg])
pc_min = pc_all.min(axis=0);  pc_max = pc_all.max(axis=0)
pc_rng = np.maximum(pc_max - pc_min, 1e-12)
pc_bg_norm = (pc_bg - pc_min) / pc_rng
pc_fg_norm = (pc_fg - pc_min) / pc_rng

xs  = np.arange(len(pc_labels))
bg_colors = cmap(norm(BG_s.astype(int)))
fg_colors = cmap(norm(P_s.astype(int)))

# Draw background lines
max_bg_lines = 2000
if len(pc_bg_norm) > max_bg_lines:
    draw_bg = rng_plot.choice(len(pc_bg_norm), max_bg_lines, replace=False)
else:
    draw_bg = np.arange(len(pc_bg_norm))

for i in draw_bg:
    ax.plot(xs, pc_bg_norm[i], color=bg_colors[i], alpha=0.06, linewidth=0.4)

# Draw Pareto lines (bold)
for i in range(len(pc_fg_norm)):
    ax.plot(xs, pc_fg_norm[i], color=fg_colors[i], alpha=0.7, linewidth=1.2)

for j in range(len(xs)):
    ax.axvline(j, color="k", linewidth=0.5, alpha=0.3)
    ax.annotate(f"{pc_min[j]:.1f}", xy=(j, -0.03), ha="center", va="top",
                fontsize=7, color="0.4")
    ax.annotate(f"{pc_max[j]:.1f}", xy=(j,  1.03), ha="center", va="bottom",
                fontsize=7, color="0.4")

ax.set_xticks(xs)
ax.set_xticklabels(pc_labels, fontsize=8)
ax.set_ylim(-0.08, 1.10)
ax.set_yticks([])
ax.set_title("Parallel coordinates — feasible manifold (faint) + Pareto front (bold)",
             fontsize=11)
_add_spring_cbar(fig, ax)
plt.tight_layout()
plt.savefig("parallel_coordinates.png", dpi=DPI)
plt.close(fig)
print("  parallel_coordinates.png")

print("\nDone.")
