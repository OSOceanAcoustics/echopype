from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

FloatSequence = Union[List[float], Tuple[float], NDArray[float]]


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


def depth_from_pressure(
    pressure: Union[float, FloatSequence],
    latitude: Optional[Union[float, FloatSequence]] = 30.0,
    atm_pres_surf: Optional[Union[float, FloatSequence]] = 0.0,
) -> NDArray[float]:
    """
    Convert pressure to depth using UNESCO 1983 algorithm.

    UNESCO. 1983. Algorithms for computation of fundamental properties of seawater (Pressure to
    Depth conversion, pages 25-27). Prepared by Fofonoff, N.P. and Millard, R.C. UNESCO technical
    papers in marine science, 44. http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    Parameters
    ----------
    pressure : Union[float, FloatSequence]
        Pressure in dbar
    latitude : Union[float, FloatSequence], default=30.0
        Latitude in decimal degrees.
    atm_pres_surf : Union[float, FloatSequence], default=0.0
        Atmospheric pressure at the surface in dbar.
        Use the default 0.0 value if pressure is corrected to be 0 at the surface.
        Otherwise, enter a correction for pressure due to air, sea ice and any other
        medium that may be present

    Returns
    -------
    depth : NDArray[float]
        Depth in meters
    """

    def _as_nparray_check(v, check_vs_pressure=False):
        """
        Convert to np.array if not already a np.array.
        Ensure latitude and atm_pres_surf are of the same size and shape as
        pressure if they are not scalar.
        """
        v_array = np.array(v) if not isinstance(v, np.ndarray) else v
        if check_vs_pressure:
            if v_array.size != 1:
                if v_array.size != pressure.size or v_array.shape != pressure.shape:
                    raise ValueError("Sequence shape or size does not match pressure")
        return v_array

    pressure = _as_nparray_check(pressure)
    latitude = _as_nparray_check(latitude, check_vs_pressure=True)
    atm_pres_surf = _as_nparray_check(atm_pres_surf, check_vs_pressure=True)

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
    depth_w_g = c1 * pressure + c2 * pressure**2 + c3 * pressure**3 + c4 * pressure**4
    x = np.sin(np.deg2rad(latitude))
    gravity = g * (1.0 + k1 * x**2 + k2 * x**4) + k3 * pressure
    depth = depth_w_g / gravity
    return depth
