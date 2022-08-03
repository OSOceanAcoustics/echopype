from parsed_to_zarr_ek60 import Parsed2ZarrEK60
import pandas as pd
from typing import List
import numpy as np


class Parsed2ZarrEK80(Parsed2ZarrEK60):
    """
    Facilitates the writing of parsed data to
    a zarr file for the EK80 sensor.
    """

    def __init__(self):
        super().__init__()

        self.power_dims = ['timestamp', 'channel_id']
        self.angle_dims = ['timestamp', 'channel_id']
        self.complex_dims = ['timestamp', 'channel_id']

    @staticmethod
    def _split_complex_data(complex_series: pd.Series) -> pd.DataFrame:
        """
        Splits the 1D complex data into two 1D arrays
        representing the real and imaginary parts of
        the complex data, for each element in ``complex_series``.

        Parameters
        ----------
        complex_series : pd.Series
            Series representing the complex data

        Returns
        -------
        DataFrame with columns backscatter_r and
        backscatter_i obtained from splitting the
        complex data into real and imaginary parts,
        respectively. The DataFrame will have the
        same index as ``complex_series``.
        """

        complex_split = complex_series.apply(
            lambda x: [np.real(x), np.imag(x)] if isinstance(x, np.ndarray) else [None, None])

        return pd.DataFrame(data=complex_split.to_list(),
                            columns=['backscatter_r', 'backscatter_i'],
                            index=complex_series.index)

    def _write_complex(self, df: pd.DataFrame, max_mb: int):
        """
        Writes the complex data and associated indices
        to a zarr group.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains angle data
        max_mb : int
            Maximum MB allowed for each chunk
        """

        # obtain complex data and drop NaNs
        complex_series = df.set_index(self.complex_dims)['complex'].dropna().copy()

        complex_df = self._split_complex_data(complex_series)

        # get unique indices
        times = complex_df.index.get_level_values(0).unique()
        channels = complex_df.index.get_level_values(1).unique()

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        complex_df = self.set_multi_index(complex_df, unique_dims)

        # write complex data to the complex group
        zarr_grp = self.zarr_root.create_group('complex')
        for column in complex_df:
            self.write_df_column(pd_series=complex_df[column], zarr_grp=zarr_grp,
                                 is_array=True, unique_time_ind=times, max_mb=max_mb)

        # write the unique indices to the complex group
        zarr_grp.array(name=self.complex_dims[0], data=times.values, dtype='<M8[ns]', fill_value='NaT')

        dtype = self._get_string_dtype(channels)
        zarr_grp.array(name=self.complex_dims[1], data=channels.values, dtype=dtype, fill_value=None)

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

        # get df corresponding to power and angle only
        pow_ang_df = datagram_df[['power', 'angle', 'timestamp', 'channel_id']].copy()

        # remove power and angle to conserve memory
        del datagram_df['power']
        del datagram_df['angle']

        # drop rows with missing power and angle data
        pow_ang_df.dropna(how='all', subset=['power', 'angle'], inplace=True)

        self._write_power(df=pow_ang_df, max_mb=max_mb)
        self._write_angle(df=pow_ang_df, max_mb=max_mb)

        self._write_complex(df=datagram_df, max_mb=max_mb)

        self._close_store()
