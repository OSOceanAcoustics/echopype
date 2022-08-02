from parsed_to_zarr import Parsed2Zarr
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

        self.power_dims = ['timestamp', 'channel']
        self.angle_dims = ['timestamp', 'channel']

    def _process_power_data(self, pd_series: pd.Series):
        """
        Applies power conversion factor to
        power data and returns it.

        Parameters
        ----------
        pd_series : pd.Series
            Series representing the power data
        """

        # Manufacturer-specific power conversion factor
        INDEX2POWER = 10.0 * np.log10(2.0) / 256.0

        return pd_series.astype("float64") * INDEX2POWER

    def _write_power(self, df: pd.DataFrame, max_mb: int) -> None:
        """
        Writes the power data and associated indices
        to a zarr group.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains power data
        max_mb : int
            Maximum MB allowed for each chunk
        """

        # obtain power data with no NA values
        power_series = df.set_index(self.power_dims)['power'].dropna().copy()

        power_series = self._process_power_data(power_series)

        # get unique indices
        times = power_series.index.get_level_values(0).unique()
        channels = power_series.index.get_level_values(1).unique()

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        power_series = self.set_multi_index(power_series, unique_dims)

        # write power data to the power group
        zarr_grp = self.zarr_root.create_group('power')
        self.write_df_column(pd_series=power_series, zarr_grp=zarr_grp,
                             is_array=True, unique_time_ind=times, max_mb=max_mb)

        # write the unique indices to the power group
        zarr_grp.array(name='timestamp', data=times, dtype='<M8[ns]', fill_value='NaN')
        zarr_grp.array(name='channel', data=channels, dtype='i8', fill_value='NaN')

    def _split_angle_data(self, angle_series):

        # split each angle element into angle_athwartship and angle_alongship
        angle_split = angle_series.apply(lambda x: [x[:, 0], x[:, 1]] if isinstance(x, np.ndarray) else [None, None])

        return pd.DataFrame(data=angle_split.to_list(),
                            columns=['angle_athwartship', 'angle_alongship'],
                            index=angle_series.index)

    def _write_angle(self, df: pd.DataFrame, max_mb: int) -> None:
        """
        Writes the angle data and associated indices
        to a zarr group.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains power data
        max_mb : int
            Maximum MB allowed for each chunk
        """

        # obtain angle data with no NA values
        angle_series = df.set_index(self.angle_dims)['angle'].dropna().copy()

        power_series = self._process_power_data(power_series)

        # get unique indices
        times = power_series.index.get_level_values(0).unique()
        channels = power_series.index.get_level_values(1).unique()

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        power_series = self.set_multi_index(power_series, unique_dims)

        # write power data to the power group
        zarr_grp = self.zarr_root.create_group('power')
        self.write_df_column(pd_series=power_series, zarr_grp=zarr_grp,
                             is_array=True, unique_time_ind=times, max_mb=max_mb)

        # write the unique indices to the power group
        zarr_grp.array(name='timestamp', data=times, dtype='<M8[ns]', fill_value='NaN')
        zarr_grp.array(name='channel', data=channels, dtype='i8', fill_value='NaN')

    def datagram_to_zarr(self, zarr_dgrams: List[dict],
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
        max_mb : int
            Maximum MB allowed for each chunk

        Notes
        -----
        This function specifically writes chunks along the time
        index.

        The chunking routine evenly distributes the times such
        that each chunk differs by at most one time. This makes
        it so that the memory required for each chunk is approximately
        the same.
        """

        datagram_df = pd.DataFrame.from_dict(zarr_dgrams)

        self._write_power(df=datagram_df, max_mb=max_mb)

        self._close_store()