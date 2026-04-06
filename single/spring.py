import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 0.8
plt.rcParams["figure.autolayout"] = True
dpi = 750
form = "png"

# Load spring data from springs.txt
springs_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "springs.txt")
springs = {}
with open(springs_file) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        springs[parts[0].lower()] = {
            "max_rotation_deg": float(parts[1]),
            "max_torque_nmm": float(parts[2]),
        }

if len(sys.argv) > 1:
    key = sys.argv[1].lower()
else:
    print("Specify a spring to load from springs.txt")
    sys.exit(1)

if key not in springs:
    print(f"Error: '{key}' not found in springs.txt")
    print(f"Available springs: {', '.join(sorted(springs.keys()))}")
    sys.exit(1)

# Spring parameters
spring_name = key
max_torque_10000_cycles_nm = springs[key]["max_torque_nmm"] / 1000.0
max_rotation_10000_cycles_deg = springs[key]["max_rotation_deg"]
power_per_degree_nm_per_deg = max_torque_10000_cycles_nm / max_rotation_10000_cycles_deg
# Integrate torque over angle with the degree-to-radian conversion.
stored_energy_j = 0.5 * power_per_degree_nm_per_deg * (max_rotation_10000_cycles_deg ** 2) * np.pi / 180.0

# 3D plot of max force transferred to the ground as a function of gear ratio and wheel diameter
n_points = 40

gear_ratios = np.linspace(20, 90, n_points)
wheel_diameters_mm = np.linspace(50, 200, n_points)
gear_ratios_mesh, wheel_diameters_mesh = np.meshgrid(gear_ratios, wheel_diameters_mm)


# vehicle parameters
vehicle_mass_kg = 1.5
gravity_m_s2 = 9.81
friction_coefficient = 0.7
safety_factor = 1.5
drivetrain_efficiency = 0.92
n_wheels = 3
n_driving_wheels = 2

target_speed_kmh = 15
target_speed_m_s = target_speed_kmh / 3.6
required_mechanical_energy_j = 0.5 * vehicle_mass_kg * (target_speed_m_s ** 2) * safety_factor
available_wheel_energy_j = stored_energy_j * drivetrain_efficiency
predicted_max_speed_m_s = np.sqrt(2 * available_wheel_energy_j / vehicle_mass_kg)
predicted_max_speed_kmh = predicted_max_speed_m_s * 3.6

gear_ratio_example = 25
wheel_diameter_example_mm = 140
wheel_radius_example_m = wheel_diameter_example_mm / 2000.0

def omega_of_t(rho, R, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Angular frequency of the spring-mass release model with lumped drivetrain
    efficiency eta applied to transmitted torque.
    """
    return np.sqrt((180.0 * eta * k_theta) / (np.pi * m * rho**2 * R**2))

def release_distance_of_t(rho, R, theta0_deg=max_rotation_10000_cycles_deg):
    """
    Vehicle travel from fully loaded spring to zero spring deflection.
    """
    return R * np.deg2rad(theta0_deg) * rho

def release_time_of_t(rho, R, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Time for the spring to unwind from theta0 to zero in the ideal model.
    """
    return np.pi / (2.0 * omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta))

def max_speed_of_t(rho, R, theta0_deg=max_rotation_10000_cycles_deg, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Peak speed reached exactly when the spring first reaches zero deflection.
    """
    return release_distance_of_t(rho, R, theta0_deg=theta0_deg) * omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta)

def time_to_speed_of_t(v_target, rho, R, theta0_deg=max_rotation_10000_cycles_deg, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Time to first reach a target speed during the release phase.
    Returns None if the target speed is not reached.
    """
    v_max = max_speed_of_t(rho, R, theta0_deg=theta0_deg, m=m, k_theta=k_theta, eta=eta)
    if v_target < 0 or v_target > v_max:
        return None

    return np.arcsin(np.clip(v_target / v_max, 0.0, 1.0)) / omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta)

def max_force_ground_of_torque(max_torque_nm, rho, R, eta=drivetrain_efficiency):
    """
    Maximum tractive force at the ground, reached at t = 0 in the ideal model.
    """
    return eta * (max_torque_nm / rho) / R

print("\n=== Spring / Vehicle Summary ===")
print(f"Spring                           : {spring_name}")
print(f"Rated max torque                 : {max_torque_10000_cycles_nm:.2f} N m")
print(f"Rated max rotation               : {max_rotation_10000_cycles_deg:.2f} deg")
print(f"Estimated spring rate            : {power_per_degree_nm_per_deg:.4f} N m/deg")
print(f"Stored spring energy             : {stored_energy_j:.2f} J")
print(f"Drivetrain efficiency            : {drivetrain_efficiency * 100.0:.2f} %")
print(f"Safety factor                    : {safety_factor:.2f}")
print(f"Available wheel energy           : {available_wheel_energy_j:.2f} J")
print(f"Vehicle mass                     : {vehicle_mass_kg:.2f} kg")
print(f"Target speed                     : {target_speed_kmh:.2f} km/h ({target_speed_m_s:.2f} m/s)")
print(f"Required energy with safety fact : {required_mechanical_energy_j:.2f} J")
print(f"Max speed with drivetrain losses : {predicted_max_speed_kmh:.2f} km/h ({predicted_max_speed_m_s:.2f} m/s)")


# Max force passed on ground total (the spring is reduuced!)
max_force_ground_n = drivetrain_efficiency * (max_torque_10000_cycles_nm / gear_ratios_mesh) / (wheel_diameters_mesh / 1000 / 2)

# Total support force on driving wheels
support_force_n = vehicle_mass_kg * gravity_m_s2 * n_driving_wheels / n_wheels

# Maximum friction force available at the driving wheels
max_friction_force = support_force_n * friction_coefficient / safety_factor
force_threshold = max_friction_force

example_peak_force_n = max_force_ground_of_torque(max_torque_10000_cycles_nm, gear_ratio_example, wheel_radius_example_m)
example_release_time_s = release_time_of_t(gear_ratio_example, wheel_radius_example_m)
example_release_distance_m = release_distance_of_t(gear_ratio_example, wheel_radius_example_m)
example_peak_speed_m_s = max_speed_of_t(gear_ratio_example, wheel_radius_example_m)
example_peak_speed_kmh = example_peak_speed_m_s * 3.6
example_peak_acceleration_m_s2 = example_peak_force_n / vehicle_mass_kg
example_time_to_target_s = time_to_speed_of_t(target_speed_m_s, gear_ratio_example, wheel_radius_example_m)
example_distance_to_target_m = None if example_time_to_target_s is None else example_release_distance_m * (1.0 - np.cos(omega_of_t(gear_ratio_example, wheel_radius_example_m) * example_time_to_target_s))
traction_margin_n = max_friction_force - example_peak_force_n
traction_margin_percent = 100.0 * traction_margin_n / max_friction_force

print(f"Driven-wheel traction limit      : {max_friction_force:.2f} N")
print("\n=== Example Drivetrain Summary ===")
print(f"Gear ratio                       : {gear_ratio_example:.2f}")
print(f"Wheel diameter                   : {wheel_diameter_example_mm:.2f} mm")
print(f"Peak wheel force at t = 0        : {example_peak_force_n:.2f} N")
print(f"Traction margin                  : {traction_margin_n:.2f} N ({traction_margin_percent:.2f}%)")
print(f"Release time                     : {example_release_time_s:.2f} s")
print(f"Release distance                 : {example_release_distance_m:.2f} m")
print(f"Peak speed at spring neutral     : {example_peak_speed_kmh:.2f} km/h ({example_peak_speed_m_s:.2f} m/s)")
print(f"Initial acceleration             : {example_peak_acceleration_m_s2:.2f} m/s^2")
if example_time_to_target_s is None:
    print(f"Target speed reach               : not reached during spring release")
else:
    print(f"Target speed reach               : {example_time_to_target_s:.2f} s at x = {example_distance_to_target_m:.2f} m")

# Masks
force_mask = np.ma.masked_less_equal(max_force_ground_n, force_threshold)
force_mask_ge = np.ma.masked_greater_equal(max_force_ground_n, force_threshold)


# 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=32, azim=37)

# Grayed out area where force is above threshold
ax.plot_surface(
    gear_ratios_mesh,
    wheel_diameters_mesh,
    force_mask,
    color='gray',
    alpha=0.15,
    rstride=1, cstride=1,
    linewidth=0, antialiased=False
)

# Main surface
surf = ax.plot_surface(
    gear_ratios_mesh, 
    wheel_diameters_mesh,
    force_mask_ge,
    # max_force_ground_n,
    cmap='viridis',
    rstride=1, cstride=1,
    linewidth=0, antialiased=True
)

# Red dashed contour line at threshold
contour = ax.contour(
    gear_ratios_mesh,
    wheel_diameters_mesh,
    max_force_ground_n,
    levels=[force_threshold],
    colors='red',
    linewidths=1.8,
    linestyles='--'
)
plt.clabel(contour, fontsize=9, fmt=f'{force_threshold:.2f} N', inline=True, colors='red')

# Labels & title
ax.set_xlabel('Gear Ratio')
ax.set_ylabel('Wheel Diameter [mm]')
ax.set_zlabel('Max Force to Ground [N]')
ax.set_title(f'Max Force to Ground vs Gear Ratio & Wheel Diameter for {spring_name}')
ax.text2D(0.02, 0.97, f'Traction limit: {force_threshold:.2f} N\nEfficiency: {drivetrain_efficiency * 100.0:.0f}%', transform=ax.transAxes)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig(f'graphs/{spring_name}_traction_surface.{form}', dpi=dpi)
plt.close(fig)

# Vehicle kinematics
def x_of_t(t, rho, R, theta0_deg=max_rotation_10000_cycles_deg, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Position x(t) for a vehicle driven by a torsional spring.
    Works with numpy arrays. This idealized expression is valid during the
    release phase 0 <= t <= pi / (2 * omega).
    """
    omega = omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta)
    x_release = release_distance_of_t(rho, R, theta0_deg=theta0_deg)
    return x_release * (1.0 - np.cos(omega * t))

def v_of_t(t, rho, R, theta0_deg=max_rotation_10000_cycles_deg, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Velocity v(t) = dx/dt during the spring release phase.
    """
    omega = omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta)
    x_release = release_distance_of_t(rho, R, theta0_deg=theta0_deg)
    return x_release * omega * np.sin(omega * t)

def a_of_t(t, rho, R, theta0_deg=max_rotation_10000_cycles_deg, m=vehicle_mass_kg, k_theta=power_per_degree_nm_per_deg, eta=drivetrain_efficiency):
    """
    Acceleration a(t) = d2x/dt2 during the spring release phase.
    """
    omega = omega_of_t(rho, R, m=m, k_theta=k_theta, eta=eta)
    x_release = release_distance_of_t(rho, R, theta0_deg=theta0_deg)
    return x_release * omega**2 * np.cos(omega * t)

# 2d plot of x(t) for gear ratio of 60 and wheel diameter of 140 mm
t_release_example = release_time_of_t(gear_ratio_example, wheel_radius_example_m)
t_array = np.linspace(0, t_release_example, 500)

fig, (ax_x, ax_v, ax_a) = plt.subplots(3, 1, sharex=True)

x_array = x_of_t(t_array, gear_ratio_example, wheel_radius_example_m)
v_array = v_of_t(t_array, gear_ratio_example, wheel_radius_example_m)
a_array = a_of_t(t_array, gear_ratio_example, wheel_radius_example_m)

ax_x.plot(t_array, x_array, color="navy")
ax_v.plot(t_array, v_array, color="orange")
ax_a.plot(t_array, a_array, color="green")

for ax in (ax_x, ax_v, ax_a):
    ax.axvline(example_release_time_s, color="black", linestyle=":", linewidth=1.0, label="Spring neutral")
    ax.grid()

ax_v.axhline(target_speed_m_s, color="red", linestyle="--", linewidth=1.0, label=f"Target speed {target_speed_kmh:.2f} km/h")

if example_time_to_target_s is not None:
    for ax in (ax_x, ax_v, ax_a):
        ax.axvline(example_time_to_target_s, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    ax_v.scatter([example_time_to_target_s], [target_speed_m_s], color="red", s=18, zorder=3)
    ax_v.annotate(
        f"Target reached at t = {example_time_to_target_s:.2f} s\nx = {example_distance_to_target_m:.2f} m",
        xy=(example_time_to_target_s, target_speed_m_s),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color="red"
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
        color="red"
    )

summary_text = "\n".join([
    f"rho (gear ratio) = {gear_ratio_example:.0f}",
    f"wheel D = {wheel_diameter_example_mm:.0f} mm",
    f"peak force = {example_peak_force_n:.2f} N",
    f"traction limit = {max_friction_force:.2f} N",
    f"release time = {example_release_time_s:.2f} s",
    f"peak speed = {example_peak_speed_kmh:.2f} km/h",
])
ax_x.text(0.02, 0.98, summary_text, transform=ax_x.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))

ax_x.set_ylabel("x [m]")
ax_v.set_ylabel("v [m/s]")
ax_a.set_ylabel("a [m/s²]")
ax_a.set_xlabel("Time [s]")
ax_x.set_title(f"Spring release response for {spring_name}")
ax_v.legend(loc="lower right")

plt.tight_layout()
plt.show()
plt.savefig(f'graphs/{spring_name}_release_response.{form}', dpi=dpi)
plt.show()
plt.close(fig)
