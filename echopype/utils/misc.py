import numpy as np


def camelcase2snakecase(camel_case_str):
    """
    Convert string from CamelCase to snake_case
    e.g. CamelCase becomes camel_case.
    """
    idx = list(reversed([i for i, c in enumerate(camel_case_str) if c.isupper()]))
    param_len = len(camel_case_str)
    for i in idx:
        #  check if we should insert an underscore
        if i > 0 and i < param_len:
            camel_case_str = camel_case_str[:i] + "_" + camel_case_str[i:]

    return camel_case_str.lower()


def depth_from_pressure(pressure, latitude=30, atm_pres_surf=0):
    """
    Convert pressure to depth using UNESCO 1983 algorithm.

    UNESCO. 1983. Algorithms for computation of fundamental properties of seawater (Pressure to
    Depth conversion, pages 25-27). Prepared by Fofonoff, N.P. and Millard, R.C. UNESCO technical
    papers in marine science, 44. http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    Parameters
    ----------
    pressure : array_like
        Pressure in dbar
    latitude : float
        Latitude in decimal degrees
    atm_pres_surf : float
        Atmospheric pressure at the surface in dbar

    Returns
    -------
    depth : array_like
        Depth in meters
    """
    # Constants
    g = 9.780318
    c1 = 9.72659
    c2 = -2.2512e-5
    c3 = 2.279e-10
    c4 = -1.82e-15
    k1 = 5.2788e-3
    k2 = 2.36e-5
    k3 = 1.092e-6

    # Calculate depth
    pressure = pressure - atm_pres_surf
    depth_w_g = c4 * pressure**4 + c3 * pressure**3 + c2 * pressure**2 + c1 * pressure
    x = np.sin(np.deg2rad(latitude))
    gravity = g * (1.0 + k1 * x**2 + k2 * x**4) + k3 * pressure
    depth = depth_w_g / gravity
    return depth
