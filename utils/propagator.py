"""
Utilities to Propagate Orbits and Uncertainties

"""
import math
import numpy as np
from utils import *
from config import re, minimum_elevation, mu_earth, simtime, iterations, h, c, M, loc, loc_hat

def accel(r) -> list:
    """
    Calculate forces on bodies
    """
    r_hat = np.zeros(3)
    f_from_earth = np.zeros(3)

    # Find magnitude and unit vector for r (Earth->Sat)
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag
    
    # Calculate force on the Sat, from Earth
    force_mag = mu_earth / (r_mag)**2
    f_from_earth = -force_mag * r_hat

    return f_from_earth

def rk4(h, v, r_sat):
    """
    Fourth-Order Runge-Kutta solver for sat position and velocity
    """
    k11 = v
    k21 = accel(r_sat)

    k12 = v + 0.5*h*k21
    k22 = accel(r_sat+0.5*h*k11)

    k13 = v + 0.5*h*k22
    k23 = accel(r_sat+0.5*h*k12)

    k14 = v + h*k23
    k24 = accel(r_sat+h*k13)

    position = r_sat + h * (k11 + 2.0*k12 + 2.0*k13 + k14) / 6.0
    velocity = v + h * (k21 + 2.0*k22 + 2.0*k23 + k24) / 6.0

    return [position, velocity]
