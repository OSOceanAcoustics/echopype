from .parsed_to_zarr_ek60 import Parsed2ZarrEK60
import pandas as pd
from typing import List
import numpy as np


class Parsed2ZarrEK80(Parsed2ZarrEK60):
    """
    Facilitates the writing of parsed data to
    a zarr file for the EK80 sensor.
    """

    def __init__(self, parser_obj):
        super().__init__(parser_obj)

        self.power_dims = ['timestamp', 'channel_id']
        self.angle_dims = ['timestamp', 'channel_id']
        self.complex_dims = ['timestamp', 'channel_id']
        self.p2z_ch_ids = {}  # channel ids for power, angle, complex

    def _get_num_transd_sec(self, x: pd.DataFrame):
        """
        Returns the number of transducer sectors.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame representing the complex series
        """

        num_transducer_sectors = np.unique(np.array(self.parser_obj.ping_data_dict["n_complex"][x.name[1]]))
        if num_transducer_sectors.size > 1:  # this is not supposed to happen
            raise ValueError("Transducer sector number changes in the middle of the file!")
        else:
            num_transducer_sectors = num_transducer_sectors[0]

        return num_transducer_sectors

    def _reshape_series(self, complex_series: pd.Series) -> pd.Series:
        """
        Reshapes complex series into the correct form, taking
        into account the beam dimension. The new shape of
        each element of ``complex_series`` will be
        (element length, num_transducer_sectors).

        Parameters
        ----------
        complex_series: pd.Series
            Series representing the complex data
        """

        # get dimension 2, which represents the number of transducer elements
        dim_2 = pd.DataFrame(complex_series).apply(self._get_num_transd_sec, axis=1)
        dim_2.name = "dim_2"

        range_sample_len = complex_series.apply(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0)

        # get dimension 1, which represents the new range_sample length
        dim_1 = (range_sample_len / dim_2).astype('int')
        dim_1.name = "dim_1"

        comp_shape_df = pd.concat([complex_series, dim_1, dim_2], axis=1)

        return comp_shape_df.apply(lambda x: x.values[0].reshape((x.dim_1, x.dim_2)) if isinstance(x.values[0], np.ndarray) else None, axis=1)

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

        # get unique indices
        times = complex_series.index.get_level_values(0).unique()
        channels = complex_series.index.get_level_values(1).unique()

        complex_series = self._reshape_series(complex_series)

        complex_df = self._split_complex_data(complex_series)

        self.p2z_ch_ids['complex'] = channels.values  # store channel ids for variable

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

        self._create_zarr_info()

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
