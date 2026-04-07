"""
Shared physics library for spring-powered vehicle analysis.

All functions take explicit parameters -- no module-level defaults.
Units: SI (m, s, kg, N internally; degrees at the interface where noted).

Physical model
--------------
A torsional spring (linear, rate k_theta in Nm/deg) unwinds through a gearbox
of ratio rho (wheel turns rho times per spring turn) onto wheels of radius R.
Drivetrain efficiency eta is modelled as a constant torque-loss factor: the
torque arriving at the wheels is eta * (spring torque / rho).

The resulting equation of motion is simple harmonic:
    theta_ddot + omega^2 * theta = 0
with omega^2 = 180 * eta * k_theta / (pi * m * rho^2 * R^2).
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_springs(filepath):
    """
    Parse springs.txt (tab-separated) and return a dict keyed by lowercase
    part number.  Torque is converted from Nmm to Nm; mass from g to kg.

    Returns
    -------
    dict : { "spf-0931": { "max_rotation_deg": float,
                            "max_torque_nm": float,
                            "mass_kg": float }, ... }
    """
    springs = {}
    with open(filepath) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            entry = {
                "max_rotation_deg": float(parts[1]),
                "max_torque_nm": float(parts[2]) / 1000.0,
            }
            if len(parts) >= 4:
                entry["mass_kg"] = float(parts[3]) / 1000.0
            else:
                entry["mass_kg"] = 0.0
            springs[parts[0].strip().lower()] = entry
    return springs


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def stored_energy(k_theta, theta0_deg):
    """
    Spring stored energy [J].

    Parameters
    ----------
    k_theta    : spring rate [Nm/deg]
    theta0_deg : maximum deflection [deg]
    """
    return 0.5 * k_theta * theta0_deg ** 2 * np.pi / 180.0


# ---------------------------------------------------------------------------
# Kinematics of the release phase
# ---------------------------------------------------------------------------

def omega(rho, R, m, k_theta, eta):
    """
    Angular frequency [rad/s] of the ideal release model.

    Parameters
    ----------
    rho     : gear ratio (wheel turns rho times per spring turn)
    R       : wheel radius [m]
    m       : vehicle mass [kg]
    k_theta : spring rate [Nm/deg]
    eta     : drivetrain efficiency [0-1]
    """
    return np.sqrt((180.0 * eta * k_theta) / (np.pi * m * rho ** 2 * R ** 2))


def release_distance(rho, R, theta0_deg):
    """
    Distance [m] the vehicle travels while the spring unwinds fully.
    Purely geometric -- independent of mass, efficiency, and spring rate.
    """
    return rho * R * np.deg2rad(theta0_deg)


def release_time(rho, R, m, k_theta, eta):
    """Time [s] for the spring to unwind from theta0 to zero."""
    return np.pi / (2.0 * omega(rho, R, m, k_theta, eta))


def max_speed(theta0_deg, m, k_theta, eta):
    """
    Peak speed [m/s] at spring neutral.
    Independent of rho and R (set entirely by energy conservation).
    """
    E = stored_energy(k_theta, theta0_deg)
    return np.sqrt(2.0 * eta * E / m)


def time_to_speed(v_target, rho, R, theta0_deg, m, k_theta, eta):
    """
    Time [s] to first reach *v_target* during the release phase.
    Returns ``np.inf`` where the target speed is not achievable.
    """
    v_max = max_speed(theta0_deg, m, k_theta, eta)
    w = omega(rho, R, m, k_theta, eta)
    ratio = v_target / v_max
    return np.where(
        ratio <= 1.0,
        np.arcsin(np.clip(ratio, 0.0, 1.0)) / w,
        np.inf,
    )


def distance_to_speed(v_target, rho, R, theta0_deg, m, k_theta, eta):
    """
    Distance [m] travelled before first reaching *v_target*.
    Returns ``np.inf`` where the target speed is not achievable.
    """
    v_max = max_speed(theta0_deg, m, k_theta, eta)
    x_rel = release_distance(rho, R, theta0_deg)
    ratio = v_target / v_max
    return np.where(
        ratio <= 1.0,
        x_rel * (1.0 - np.sqrt(np.maximum(1.0 - ratio ** 2, 0.0))),
        np.inf,
    )


def position(t, rho, R, theta0_deg, m, k_theta, eta):
    """
    Position x(t) [m] during the release phase.
    Valid for 0 <= t <= release_time(...).
    """
    w = omega(rho, R, m, k_theta, eta)
    x_rel = release_distance(rho, R, theta0_deg)
    return x_rel * (1.0 - np.cos(w * t))


def velocity(t, rho, R, theta0_deg, m, k_theta, eta):
    """Velocity v(t) [m/s] during the release phase."""
    w = omega(rho, R, m, k_theta, eta)
    x_rel = release_distance(rho, R, theta0_deg)
    return x_rel * w * np.sin(w * t)


def acceleration(t, rho, R, theta0_deg, m, k_theta, eta):
    """Acceleration a(t) [m/s^2] during the release phase."""
    w = omega(rho, R, m, k_theta, eta)
    x_rel = release_distance(rho, R, theta0_deg)
    return x_rel * w ** 2 * np.cos(w * t)


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------

def max_force_ground(max_torque_nm, rho, R, eta):
    """Peak tractive force [N] at the wheel contact patch (t = 0)."""
    return eta * max_torque_nm / (rho * R)


def traction_limit(m, g, n_driving, n_total, mu, safety_factor):
    """Maximum friction force [N] available at the driving wheels."""
    return mu * m * g * n_driving / (n_total * safety_factor)
