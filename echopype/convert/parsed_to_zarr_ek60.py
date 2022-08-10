import numpy as np
import pandas as pd
import psutil

from .parsed_to_zarr import Parsed2Zarr


class Parsed2ZarrEK60(Parsed2Zarr):
    """
    Facilitates the writing of parsed data to
    a zarr file for the EK60 sensor.
    """

    def __init__(self, parser_obj):
        super().__init__(parser_obj)

        self.power_dims = ["timestamp", "channel"]
        self.angle_dims = ["timestamp", "channel"]
        self.p2z_ch_ids = {}  # channel ids for power, angle, complex
        self.datagram_df = None  # df created from zarr variables

    @staticmethod
    def _get_string_dtype(pd_series: pd.Index) -> str:
        """
        Returns the string dtype in a format that
        works for zarr.

        Parameters
        ----------
        pd_series: pd.Index
            A series where all of the elements are strings
        """

        if all(pd_series.map(type) == str):
            max_len = pd_series.map(len).max()
            dtype = f"<U{max_len}"
        else:
            raise ValueError("All elements of pd_series must be strings!")

        return dtype

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

        # obtain power data
        power_series = df.set_index(self.power_dims)["power"].copy()

        # get unique indices
        times = power_series.index.get_level_values(0).unique()
        channels = power_series.index.get_level_values(1).unique()

        self.p2z_ch_ids["power"] = channels.values  # store channel ids for variable

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        power_series = self.set_multi_index(power_series, unique_dims)

        # write power data to the power group
        zarr_grp = self.zarr_root.create_group("power")
        self.write_df_column(
            pd_series=power_series,
            zarr_grp=zarr_grp,
            is_array=True,
            unique_time_ind=times,
            max_mb=max_mb,
        )

        # write the unique indices to the power group
        zarr_grp.array(
            name=self.power_dims[0], data=times.values, dtype=times.dtype.str, fill_value="NaT"
        )

        dtype = self._get_string_dtype(channels)
        zarr_grp.array(name=self.power_dims[1], data=channels.values, dtype=dtype, fill_value=None)

    @staticmethod
    def _split_angle_data(angle_series: pd.Series) -> pd.DataFrame:
        """
        Splits the 2D angle data into two 1D arrays
        representing angle_athwartship and angle_alongship,
        for each element in ``angle_series``.

        Parameters
        ----------
        angle_series : pd.Series
            Series representing the angle data

        Returns
        -------
        DataFrame with columns angle_athwartship and
        angle_alongship obtained from splitting the
        2D angle data, with that same index as
        ``angle_series``
        """

        # split each angle element into angle_athwartship and angle_alongship
        angle_split = angle_series.apply(
            lambda x: [x[:, 0], x[:, 1]] if isinstance(x, np.ndarray) else [None, None]
        )

        return pd.DataFrame(
            data=angle_split.to_list(),
            columns=["angle_athwartship", "angle_alongship"],
            index=angle_series.index,
        )

    def _write_angle(self, df: pd.DataFrame, max_mb: int) -> None:
        """
        Writes the angle data and associated indices
        to a zarr group.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains angle data
        max_mb : int
            Maximum MB allowed for each chunk
        """

        # obtain angle data
        angle_series = df.set_index(self.angle_dims)["angle"].copy()

        angle_df = self._split_angle_data(angle_series)

        # get unique indices
        times = angle_df.index.get_level_values(0).unique()
        channels = angle_df.index.get_level_values(1).unique()

        self.p2z_ch_ids["angle"] = channels.values  # store channel ids for variable

        # create multi index using the product of the unique dims
        unique_dims = [times, channels]

        angle_df = self.set_multi_index(angle_df, unique_dims)

        # write angle data to the angle group
        zarr_grp = self.zarr_root.create_group("angle")
        for column in angle_df:
            self.write_df_column(
                pd_series=angle_df[column],
                zarr_grp=zarr_grp,
                is_array=True,
                unique_time_ind=times,
                max_mb=max_mb,
            )

        # write the unique indices to the angle group
        zarr_grp.array(
            name=self.angle_dims[0], data=times.values, dtype=times.dtype.str, fill_value="NaT"
        )

        dtype = self._get_string_dtype(channels)
        zarr_grp.array(name=self.angle_dims[1], data=channels.values, dtype=dtype, fill_value=None)

    def _get_power_angle_size(self, df: pd.DataFrame) -> int:
        """
        Returns the total memory in bytes required to
        store the expanded power and angle data.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the power, angle, and
            the appropriate dimension data
        """

        # get unique indices
        times = df[self.power_dims[0]].unique()
        channels = df[self.power_dims[1]].unique()

        # get final form of index
        multi_index = pd.MultiIndex.from_product([times, channels])

        # get the total memory required for expanded zarr variables
        pow_mem = self.array_series_bytes(df["power"], multi_index.shape[0])
        angle_mem = self.array_series_bytes(df["angle"], multi_index.shape[0])

        return pow_mem + angle_mem

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

        # create datagram df, if it does not exist
        if not isinstance(self.datagram_df, pd.DataFrame):
            self.datagram_df = pd.DataFrame.from_dict(self.parser_obj.zarr_datagrams)

        total_mem = self._get_power_angle_size(self.datagram_df)

        # get statistics about system memory usage
        mem = psutil.virtual_memory()

        zarr_dgram_size = self._get_zarr_dgrams_size()

        # approx. the amount of memory that will be used after expansion
        req_mem = mem.used - zarr_dgram_size + total_mem

        # free memory, if we no longer need it
        if mem.total * mem_mult > req_mem:
            del self.datagram_df
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

        # create datagram df, if it does not exist
        if not isinstance(self.datagram_df, pd.DataFrame):
            self.datagram_df = pd.DataFrame.from_dict(self.parser_obj.zarr_datagrams)
            del self.parser_obj.zarr_datagrams  # free memory

        # convert channel column to a string
        self.datagram_df["channel"] = self.datagram_df["channel"].astype(str)

        self._write_power(df=self.datagram_df, max_mb=max_mb)

        del self.datagram_df["power"]  # free memory

        self._write_angle(df=self.datagram_df, max_mb=max_mb)

        del self.datagram_df  # free memory

        self._close_store()
