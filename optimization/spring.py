#!/usr/bin/env python3
"""
Analyze one spring-powered vehicle configuration and plot x(t), v(t), and a(t).

Usage:
    python spring.py -r <ratio> -d <diameter_mm> -m <mass_kg> -s <spring>
"""

import os
import json
import argparse

GREEN = "\x1b[32m"
RED   = "\x1b[31m"
RESET = "\x1b[0m"

def ok(cond, fail_msg=""):
    return f"{GREEN}YES{RESET}" if cond else f"{RED}NO  <-- {fail_msg}{RESET}"

import numpy as np
import matplotlib

import spring_model as sm


parser = argparse.ArgumentParser(
    description="Analyze a single spring-powered vehicle configuration.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-r", "--ratio",
    type=float, required=True, dest="gear_ratio",
    help="Gear ratio (rho)",
)
parser.add_argument(
    "-d", "--diameter",
    type=float, required=True, dest="wheel_diameter_mm",
    help="Wheel diameter [mm]",
)
parser.add_argument(
    "-m", "--mass",
    type=float, required=True,
    help="Vehicle mass [kg] (excluding spring mass)",
)
parser.add_argument(
    "-s", "--spring",
    type=str, required=True,
    help="Spring part number (e.g. SPF-0927)",
)
parser.add_argument(
    "-i", "--interactive",
    action="store_true",
    help="Show plots interactively instead of only saving them",
)
parser.add_argument(
    "-f", "--file",
    default="springs.txt",
    metavar="SPRINGS_FILE",
    help="Spring data file to load",
)
args = parser.parse_args()

if not args.interactive:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
os.makedirs(GRAPHS_DIR, exist_ok=True)

with open(CONFIG_FILE) as fh:
    config = json.load(fh)

physics_config = config["physics"]

if os.path.isabs(args.file):
    springs_file = args.file
else:
    springs_file = os.path.join(SCRIPT_DIR, args.file)

springs = sm.load_springs(springs_file)
spring_name = args.spring.lower()

if spring_name not in springs:
    available = ", ".join(sorted(springs))
    raise SystemExit(
        f"Error: '{spring_name}' not found in {springs_file}\nAvailable springs: {available}"
    )

if args.gear_ratio <= 0.0:
    raise SystemExit("Error: gear_ratio must be positive")
if args.wheel_diameter_mm <= 0.0:
    raise SystemExit("Error: wheel_diameter_mm must be positive")
if args.mass <= 0.0:
    raise SystemExit("Error: mass must be positive")

spring = springs[spring_name]
target_speed_kmh = physics_config["target_speed_kmh"]
target_speed_ms = target_speed_kmh / 3.6
friction_coeff = physics_config["friction_coeff"]
safety_factor = physics_config["safety_factor"]
drivetrain_eff = physics_config["drivetrain_eff"]
n_wheels = physics_config["n_wheels"]
n_driving_wheels = physics_config["n_driving_wheels"]
gravity = physics_config["gravity"]
dpi = physics_config["dpi"]

vehicle_mass_kg = args.mass
spring_mass_kg = spring["mass_kg"]
total_mass_kg = vehicle_mass_kg + spring_mass_kg
gear_ratio = args.gear_ratio
wheel_diameter_mm = args.wheel_diameter_mm
wheel_radius_m = wheel_diameter_mm / 2000.0
theta0_deg = spring["max_rotation_deg"]
max_torque_nm = spring["max_torque_nm"]
k_theta = max_torque_nm / theta0_deg

stored_energy_j = sm.stored_energy(k_theta, theta0_deg)
available_wheel_energy_j = drivetrain_eff * stored_energy_j
required_mechanical_energy_j = 0.5 * total_mass_kg * target_speed_ms ** 2 * safety_factor

peak_force_n = sm.max_force_ground(max_torque_nm, gear_ratio, wheel_radius_m, drivetrain_eff)
max_friction_force_n = sm.traction_limit(
    total_mass_kg,
    gravity,
    n_driving_wheels,
    n_wheels,
    friction_coeff,
    safety_factor,
)
traction_margin_n = max_friction_force_n - peak_force_n
traction_margin_percent = 100.0 * traction_margin_n / max_friction_force_n

release_time_s = sm.release_time(gear_ratio, wheel_radius_m, total_mass_kg, k_theta, drivetrain_eff)
release_distance_m = sm.release_distance(gear_ratio, wheel_radius_m, theta0_deg)
peak_speed_m_s = sm.max_speed(theta0_deg, total_mass_kg, k_theta, drivetrain_eff)
peak_speed_kmh = peak_speed_m_s * 3.6
initial_acceleration_m_s2 = peak_force_n / total_mass_kg
omega_rad_s = sm.omega(gear_ratio, wheel_radius_m, total_mass_kg, k_theta, drivetrain_eff)

time_to_target_s = sm.time_to_speed(
    target_speed_ms,
    gear_ratio,
    wheel_radius_m,
    theta0_deg,
    total_mass_kg,
    k_theta,
    drivetrain_eff,
)
distance_to_target_m = sm.distance_to_speed(
    target_speed_ms,
    gear_ratio,
    wheel_radius_m,
    theta0_deg,
    total_mass_kg,
    k_theta,
    drivetrain_eff,
)

target_reached = bool(np.isfinite(time_to_target_s))

# Compute spring turns to reach target speed
# theta(t) = theta0 * cos(omega*t), so unwound angle = theta0*(1 - cos(omega*t))
if target_reached:
    theta_unwound_at_target_deg = theta0_deg * (1.0 - np.cos(omega_rad_s * time_to_target_s))
    turns_to_target = theta_unwound_at_target_deg / 360.0
else:
    theta_unwound_at_target_deg = np.nan
    turns_to_target = np.nan

total_spring_turns = theta0_deg / 360.0
traction_ok = peak_force_n <= max_friction_force_n
energy_ok = available_wheel_energy_j >= required_mechanical_energy_j

W = 74
SEP = "=" * W

print()
print(SEP)
print(f"  Configuration test: {spring_name.upper()}  |  ratio {gear_ratio:g}  |  D {wheel_diameter_mm:g} mm  |  m {vehicle_mass_kg:g} kg")
print(SEP)

print("\n--- Spring ---")
print(f"  Rated max torque       : {max_torque_nm:.4f} N m")
print(f"  Rated max rotation     : {theta0_deg:.2f} deg")
print(f"  Spring rate            : {k_theta:.6f} N m/deg")
print(f"  Stored energy          : {stored_energy_j:.4f} J")
print(f"  Spring mass            : {spring_mass_kg:.4f} kg")

print("\n--- Mass budget ---")
print(f"  Vehicle mass           : {vehicle_mass_kg:.4f} kg")
print(f"  Spring mass            : {spring_mass_kg:.4f} kg")
print(f"  Total accelerated mass : {total_mass_kg:.4f} kg")

print("\n--- Drivetrain ---")
print(f"  Gear ratio             : {gear_ratio:.4f}")
print(f"  Wheel diameter         : {wheel_diameter_mm:.2f} mm")
print(f"  Wheel radius           : {wheel_radius_m:.4f} m")
print(f"  Drivetrain efficiency  : {drivetrain_eff * 100.0:.2f} %")
print(f"  Natural frequency      : {omega_rad_s:.4f} rad/s")

print("\n--- Kinematics ---")
print(f"  Release time           : {release_time_s:.4f} s")
print(f"  Release distance       : {release_distance_m:.4f} m")
print(f"  Peak speed             : {peak_speed_kmh:.4f} km/h [{peak_speed_m_s:.4f} m/s] (includes efficiency, not safety factor)")
print(f"  Initial acceleration   : {initial_acceleration_m_s2:.4f} m/s^2 [{(initial_acceleration_m_s2 / 9.81):.2f} g]")
if target_reached:
    print(f"  Time to {target_speed_kmh:.1f} km/h      : {time_to_target_s:.4f} s at x = {distance_to_target_m:.4f} m")
    print(f"  Turns to {target_speed_kmh:.1f} km/h     : {turns_to_target:.4f} turns (of {total_spring_turns:.4f} total)")
else:
    print(f"  Time to {target_speed_kmh:.1f} km/h      : not reached during spring release")

print(f"\n--- Constraints  (safety factor = {safety_factor}) ---")
print(f"  [energy]  available    : {available_wheel_energy_j:.4f} J   (multiplied by drivetrain efficiency)")
print(f"  [energy]  required     : {required_mechanical_energy_j:.4f} J   (multiplied by safety factor)")
print(f"  [energy]  satisfied    : {ok(energy_ok, 'INSUFFICIENT ENERGY')}")
print(f"  [traction] peak force  : {peak_force_n:.4f} N   (at t = 0)")
print(f"  [traction] limit       : {max_friction_force_n:.4f} N   (slip threshold with safety factor)")
print(f"  [traction] margin      : {traction_margin_n:.4f} N   ({traction_margin_percent:.2f}%)")
print(f"  [traction] satisfied   : {ok(traction_ok, 'WHEEL SLIP')}")

print()
print(SEP)

t_array = np.linspace(0.0, release_time_s, 500)
x_array = sm.position(t_array, gear_ratio, wheel_radius_m, theta0_deg, total_mass_kg, k_theta, drivetrain_eff)
v_array = sm.velocity(t_array, gear_ratio, wheel_radius_m, theta0_deg, total_mass_kg, k_theta, drivetrain_eff)
a_array = sm.acceleration(t_array, gear_ratio, wheel_radius_m, theta0_deg, total_mass_kg, k_theta, drivetrain_eff)

fig, (ax_x, ax_v, ax_a) = plt.subplots(3, 1, sharex=True, figsize=(18, 10))

ax_x.plot(t_array, x_array, color="navy")
ax_v.plot(t_array, v_array, color="orange")
ax_a.plot(t_array, a_array, color="green")

for ax in (ax_x, ax_v, ax_a):
    ax.axvline(release_time_s, color="black", linestyle=":", linewidth=2.0, label="Spring neutral")
    ax.grid()

ax_v.axhline(
    target_speed_ms,
    color="red",
    linestyle="--",
    linewidth=1.0,
    label=f"Target speed {target_speed_kmh:.2f} km/h",
)

if target_reached:
    for ax in (ax_x, ax_v, ax_a):
        ax.axvline(time_to_target_s, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    ax_v.scatter([time_to_target_s], [target_speed_ms], color="red", s=18, zorder=3)
    ax_v.annotate(
        f"Target v reached at t = {time_to_target_s:.2f} s\nx = {distance_to_target_m:.2f} m\nspring turns = {turns_to_target:.2f} / {total_spring_turns:.2f}",
        xy=(time_to_target_s, target_speed_ms),
        xytext=(-10, 10),
        textcoords="offset points",
        fontsize=8,
        color="red",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="red", boxstyle="round,pad=0.4"),
    )
else:
    ax_v.text(
        0.98,
        0.08,
        "Target speed not reached\nduring spring release",
        transform=ax_v.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="red",
    )

summary_text = "\n".join([
    f"gear ratio = {gear_ratio:.2f}",
    f"wheel diameter = {wheel_diameter_mm:.0f} mm",
    f"total mass = {total_mass_kg:.2f} kg",
    f"release time = {release_time_s:.2f} s",
    f"peak speed = {peak_speed_kmh:.2f} km/h",
])
ax_x.text(
    0.02,
    0.90,
    summary_text,
    transform=ax_x.transAxes,
    va="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8", boxstyle="round,pad=0.4"),
)

ax_x.set_ylabel("x [m]")
ax_v.set_ylabel("v [m/s]")
ax_a.set_ylabel("a [m/s^2]")
ax_a.set_xlabel("Time [s]")
ax_x.set_title(f"Spring release response for {spring_name}")
ax_v.legend(loc="lower right")

output_name = (
    f"{spring_name}_rho-{gear_ratio:g}_diam-{wheel_diameter_mm:g}mm_"
    f"mass-{vehicle_mass_kg:g}kg_release_response.png"
)
output_path = os.path.join(GRAPHS_DIR, output_name)

plt.tight_layout()
plt.savefig(output_path, dpi=dpi)
if args.interactive:
    plt.show()
plt.close(fig)

print(f"\nSaved graph: {output_name}")
