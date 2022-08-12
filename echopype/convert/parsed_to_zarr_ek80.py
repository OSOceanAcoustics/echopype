import numpy as np
import pandas as pd
import psutil

from .parsed_to_zarr_ek60 import Parsed2ZarrEK60


class Parsed2ZarrEK80(Parsed2ZarrEK60):
    """
    Facilitates the writing of parsed data to
    a zarr file for the EK80 sensor.
    """

    def __init__(self, parser_obj):
        super().__init__(parser_obj)

        self.power_dims = ["timestamp", "channel_id"]
        self.angle_dims = ["timestamp", "channel_id"]
        self.complex_dims = ["timestamp", "channel_id"]
        self.p2z_ch_ids = {}  # channel ids for power, angle, complex
        self.pow_ang_df = None  # df that holds power and angle data
        self.complex_df = None  # df that holds complex data

    def _get_num_transd_sec(self, x: pd.DataFrame):
        """
        Returns the number of transducer sectors.

        Parameters
        ----------
        x : pd.DataFrame
            DataFrame representing the complex series
        """

        num_transducer_sectors = np.unique(
            np.array(self.parser_obj.ping_data_dict["n_complex"][x.name[1]])
        )
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

        range_sample_len = complex_series.apply(
            lambda x: x.shape[0] if isinstance(x, np.ndarray) else 0
        )

        # get dimension 1, which represents the new range_sample length
        dim_1 = (range_sample_len / dim_2).astype("int")
        dim_1.name = "dim_1"

        comp_shape_df = pd.concat([complex_series, dim_1, dim_2], axis=1)

        return comp_shape_df.apply(
            lambda x: x.values[0].reshape((x.dim_1, x.dim_2))
            if isinstance(x.values[0], np.ndarray)
            else None,
            axis=1,
        )

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
            lambda x: [np.real(x), np.imag(x)] if isinstance(x, np.ndarray) else [None, None]
        )

        return pd.DataFrame(
            data=complex_split.to_list(),
            columns=["backscatter_r", "backscatter_i"],
            index=complex_series.index,
        )

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
        complex_series = df.set_index(self.complex_dims)["complex"].copy()

        # get unique indices
        times = complex_series.index.get_level_values(0).unique()
        channels = complex_series.index.get_level_values(1).unique()

        complex_series = self._reshape_series(complex_series)

        complex_df = self._split_complex_data(complex_series)

        self.p2z_ch_ids["complex"] = channels.values  # store channel ids for variable

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        complex_df = self.set_multi_index(complex_df, unique_dims)

        # write complex data to the complex group
        zarr_grp = self.zarr_root.create_group("complex")
        for column in complex_df:
            self.write_df_column(
                pd_series=complex_df[column],
                zarr_grp=zarr_grp,
                is_array=True,
                unique_time_ind=times,
                max_mb=max_mb,
            )

        # write the unique indices to the complex group
        zarr_grp.array(
            name=self.complex_dims[0], data=times.values, dtype=times.dtype.str, fill_value="NaT"
        )

        dtype = self._get_string_dtype(channels)
        zarr_grp.array(
            name=self.complex_dims[1], data=channels.values, dtype=dtype, fill_value=None
        )

    def _get_complex_size(self, df: pd.DataFrame) -> int:
        """
        Returns the total memory in bytes required to
        store the expanded complex data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the complex and
            the appropriate dimension data
        """

        # get unique indices
        times = df[self.complex_dims[0]].unique()
        channels = df[self.complex_dims[1]].unique()

        # get final form of index
        multi_index = pd.MultiIndex.from_product([times, channels])

        # get the total memory required for expanded zarr variables
        complex_mem = self.array_series_bytes(df["complex"], multi_index.shape[0])

        # multiply by 2 because we store both the complex and real parts
        return 2 * complex_mem

    def _get_zarr_dfs(self):
        """
        Creates the DataFrames that hold the power, angle, and
        complex data, which are needed for downstream computation.
        """

        datagram_df = pd.DataFrame.from_dict(self.parser_obj.zarr_datagrams)

        # get df corresponding to power and angle only
        self.pow_ang_df = datagram_df[["power", "angle", "timestamp", "channel_id"]].copy()

        # remove power and angle to conserve memory
        del datagram_df["power"]
        del datagram_df["angle"]

        # drop rows with missing power and angle data
        self.pow_ang_df.dropna(how="all", subset=["power", "angle"], inplace=True)

        self.complex_df = datagram_df.dropna().copy()

    def write_to_zarr(self, mem_mult: float = 0.3) -> bool:
        """
        Determines if the zarr data provided will expand
        into a form that is larger than a percentage of
        the total physical RAM.

        Parameters
        ----------
        mem_mult : float
            Multiplier for total physical RAM

        Notes
        -----
        If ``mem_mult`` times the total RAM is less
        than the total memory required to store the
        expanded zarr variables, this function will
        return True, otherwise False.
        """
        isinstance(self.datagram_df, pd.DataFrame)
        # create zarr dfs, if they do not exist
        if not isinstance(self.pow_ang_df, pd.DataFrame) and not isinstance(
            self.complex_df, pd.DataFrame
        ):
            self._get_zarr_dfs()

        # get memory required for zarr data
        pow_ang_total_mem = self._get_power_angle_size(self.pow_ang_df)
        comp_total_mem = self._get_complex_size(self.complex_df)
        total_mem = pow_ang_total_mem + comp_total_mem

        # get statistics about system memory usage
        mem = psutil.virtual_memory()

        zarr_dgram_size = self._get_zarr_dgrams_size()

        # approx. the amount of memory that will be used after expansion
        req_mem = mem.used - zarr_dgram_size + total_mem

        # free memory, if we no longer need it
        if mem.total * mem_mult > req_mem:
            del self.pow_ang_df
            del self.complex_df
        else:
            del self.parser_obj.zarr_datagrams

        return mem.total * mem_mult < req_mem

    def datagram_to_zarr(self, max_mb: int) -> None:
        """
        Facilitates the conversion of a list of
        datagrams to a form that can be written
        to a zarr store.

        Parameters
        ----------
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

        # create zarr dfs, if they do not exist
        if not isinstance(self.pow_ang_df, pd.DataFrame) and not isinstance(
            self.complex_df, pd.DataFrame
        ):
            self._get_zarr_dfs()
            del self.parser_obj.zarr_datagrams  # free memory

        self._write_power(df=self.pow_ang_df, max_mb=max_mb)
        self._write_angle(df=self.pow_ang_df, max_mb=max_mb)

        del self.pow_ang_df  # free memory

        self._write_complex(df=self.complex_df, max_mb=max_mb)

        del self.complex_df  # free memory

        self._close_store()
