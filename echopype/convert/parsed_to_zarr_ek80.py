from .parsed_to_zarr import Parsed2Zarr
import pandas as pd
from typing import List
import numpy as np


class Parsed2ZarrEK60(Parsed2Zarr):
    """
    Facilitates the writing of parsed data to
    a zarr file for the EK60 sensor.
    """

    def __init__(self):
        super().__init__()

    def _process_power_data(self, reduced_datagram):

        if "power" in reduced_datagram.keys():

            if ("ALL" in self.data_type) and isinstance(reduced_datagram["power"], np.ndarray):
                # Manufacturer-specific power conversion factor
                INDEX2POWER = 10.0 * np.log10(2.0) / 256.0

                reduced_datagram["power"] = reduced_datagram["power"].astype("float32") * INDEX2POWER

    def _split_angle_data(self, angle_val):

        # account for None values
        if isinstance(angle_val, np.ndarray):
            angle_split = {"angle_athwartship": angle_val[:, 0],
                           "angle_alongship": angle_val[:, 1]}
        else:
            angle_split = {"angle_athwartship": None,
                           "angle_alongship": None}

    def _split_complex_data(self, complex_val):

        # account for None values
        if isinstance(complex_val, np.ndarray):
            complex_split = {"backscatter_r": np.real(complex_val),
                             "backscatter_i": np.imag(complex_val)}
        else:
            complex_split = {"backscatter_r": None,
                             "backscatter_i": None}



    def datagram_to_zarr(self, zarr_dgrams: List[dict], zarr_vars: dict,
                         max_mb: int) -> None:
        """
        Facilitates the conversion of a list of
        datagrams to a form that can be written
        to a zarr store.

        Parameters
        ----------
        zarr_dgrams : List[dict]
            A list of datagrams where each datagram contains
            at least one variable that should be written to
            a zarr file and any associated dimensions.
        zarr_vars : dict
            A dictionary where the keys represent the variable
            that should be written to a zarr file and the values
            are a list of the variable's dimensions.
        max_mb : int
            Maximum MB allowed for each chunk

        Notes
        -----
        This function specifically writes chunks along the time
        index.

        The dimensions provided in ``zarr_vars`` must have the
        time dimension as the first element.

        The chunking routine evenly distributes the times such
        that each chunk differs by at most one time. This makes
        it so that the memory required for each chunk is approximately
        the same.
        """

        datagram_df = pd.DataFrame.from_dict(zarr_dgrams)

        # unique_dims = map(list, set(map(tuple, zarr_vars.values())))
        #
        # # write groups of variables with the same dimensions to zarr
        # for dims in unique_dims:
        #     # get all variables with dimensions dims
        #     var_names = [key for key, val in zarr_vars.items() if val == dims]
        #
        #     # columns needed to compute df_multi
        #     req_cols = var_names + dims
        #
        #     df_multi = set_multi_index(datagram_df[req_cols], dims)
        #
        #     # check to make sure the second index is unique
        #     unique_second_index(df_multi)
        #
        #     write_df_to_zarr(df_multi, array_grp, time_name=dims[0], max_mb=max_mb)

        self._close_store()