import xarray as xr
import numpy as np
import pandas as pd
from echopype.metrics.summary_statistics import (
    delta_z,
    convert_to_linear,
    abundance,
    center_of_mass,
    dispersion,
    evenness,
    aggregation,
)


# Utility Function


def create_test_ds(Sv, range):
    freq = [30]
    time = pd.date_range("2021-08-28", periods=2)
    reference_time = pd.Timestamp("2021-08-27")
    r_b = [0, 1, 2]

    testDS = xr.Dataset(
        data_vars=dict(
            Sv=(["frequency", "ping_time", "range_bin"], Sv),
            range=(["frequency", "ping_time", "range_bin"], range),
        ),
        coords={
            'frequency': xr.DataArray(
                freq,
                name='frequency',
                dims=['frequency'],
                attrs={'units': 'kHz'},
            ),
            'ping_time': xr.DataArray(
                time,
                name='ping_time',
                dims=['ping_time'],
            ),
            'range_bin': xr.DataArray(
                r_b,
                name='range_bin',
                dims=['range_bin'],
            ),
        },
    )
    return testDS


# Test Functions


def test_abundance():
    """Compares summary_statistics.py calculation of abundance with verified outcomes"""
    Sv = np.array([[[20, 40, 60], [50, 20, 30]]])
    range = np.array([[[1, 2, 3], [2, 3, 4]]])
    
    ab_ds1 = create_test_ds(Sv, range)
    ab_ds1_SOL = np.array([[60.04321374, 30.41392685]])
    assert np.allclose(
        abundance(ab_ds1), ab_ds1_SOL, rtol=1e-09
    ), 'Calculated output does not match expected output'


def test_center_of_mass():
    """Compares summary_statistics.py calculation of center_of_mass with verified outcomes"""
    Sv = np.array([[[20, 40, 60], [50, 20, 30]]])
    range = np.array([[[1, 2, 3], [2, 3, 4]]])
    cm_ds1 = create_test_ds(Sv, range)
    cm_ds1_SOL = np.array([[2.99009901, 3.90909090]])
    assert np.allclose(
        center_of_mass(cm_ds1), cm_ds1_SOL, rtol=1e-09
    ), 'Calculated output does not match expected output'


def test_inertia():
    """Compares summary_statistics.py calculation of intertia with verified outcomes"""
    Sv = np.array([[[20, 40, 60], [50, 20, 30]]])
    range = np.array([[[1, 2, 3], [2, 3, 4]]])
    in_ds1 = create_test_ds(Sv, range)
    in_ds1_SOL = np.array([[0.00980296, 0.08264463]])
    assert np.allclose(
        dispersion(in_ds1), in_ds1_SOL, rtol=1e-09
    ), 'Calculated output does not match expected output'


def test_evenness():
    """Compares summary_statistics.py calculation of evenness with verified outcomes"""
    Sv = np.array([[[20, 40, 60], [50, 20, 30]]])
    range = np.array([[[1, 2, 3], [2, 3, 4]]])
    ev_ds1 = create_test_ds(Sv, range)
    ev_ds1_SOL = np.array([[1.019998, 1.198019802]])
    assert np.allclose(
        evenness(ev_ds1), ev_ds1_SOL, rtol=1e-09
    ), 'Calculated output does not match expected output'


def test_aggregation():
    """Compares summary_statistics.py calculation of aggregation with verified outcomes"""
    Sv = np.array([[[20, 40, 60], [50, 20, 30]]])
    range = np.array([[[1, 2, 3], [2, 3, 4]]])
    ag_ds1 = create_test_ds(Sv, range)
    ag_ds1_SOL = np.array([[0.9803940792, 0.8347107438]])
    assert np.allclose(
        aggregation(ag_ds1), ag_ds1_SOL, rtol=1e-09
    ), 'Calculated output does not match expected output'
