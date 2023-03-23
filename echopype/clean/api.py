"""
Functions for reducing variabilities in backscatter data.
"""

from ..utils.prov import add_processing_level, echopype_prov_attrs, insert_input_processing_level
from .noise_est import NoiseEst


def estimate_noise(ds_Sv, ping_num, range_sample_num, noise_max=None):
    """
    Estimate background noise by computing mean calibrated power of a collection of pings.

    See ``remove_noise`` for reference.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions

    Returns
    -------
    A DataArray containing noise estimated from the input ``ds_Sv``
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.estimate_noise(noise_max=noise_max)
    return noise_obj.Sv_noise


@add_processing_level("L*B")
def remove_noise(ds_Sv, ping_num, range_sample_num, noise_max=None, SNR_threshold=3):
    """
    Remove noise by using estimates of background noise
    from mean calibrated power of a collection of pings.

    Reference: De Robertis & Higginbottom. 2007.
    A post-processing technique to estimate the signal-to-noise ratio
    and remove echosounder background noise.
    ICES Journal of Marine Sciences 64(6): 1282–1291.

    Parameters
    ----------
    ds_Sv : xr.Dataset
        dataset containing ``Sv`` and ``echo_range`` [m]
    ping_num : int
        number of pings to obtain noise estimates
    range_sample_num : int
        number of samples along the ``range_sample`` dimension to obtain noise estimates
    noise_max : float
        the upper limit for background noise expected under the operating conditions
    SNR_threshold : float
        acceptable signal-to-noise ratio, default to 3 dB

    Returns
    -------
    The input dataset with additional variables, including
    the corrected Sv (``Sv_corrected``) and the noise estimates (``Sv_noise``)
    """
    noise_obj = NoiseEst(ds_Sv=ds_Sv.copy(), ping_num=ping_num, range_sample_num=range_sample_num)
    noise_obj.remove_noise(noise_max=noise_max, SNR_threshold=SNR_threshold)
    ds_Sv = noise_obj.ds_Sv

    prov_dict = echopype_prov_attrs(process_type="processing")
    prov_dict["processing_function"] = "clean.remove_noise"
    ds_Sv = ds_Sv.assign_attrs(prov_dict)

    # The output ds_Sv is built as a copy of the input ds_Sv, so the step below is
    # not needed, strictly speaking. But doing makes the decorator function more generic
    ds_Sv = insert_input_processing_level(ds_Sv, input_ds=ds_Sv)

    return ds_Sv
