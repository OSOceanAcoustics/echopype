import pytest
import xarray as xr
import pandas as pd
from typing import List, Optional, Tuple
from echopype import open_raw
import shutil
from zarr.hierarchy import Group as ZGroup
import os.path
from fsspec import FSMap
from s3fs import S3FileSystem
import requests
import time
from echopype.convert.parsed_to_zarr import Parsed2Zarr, DEFAULT_ZARR_TEMP_DIR
from echopype.convert.parsed_to_zarr_ek60 import Parsed2ZarrEK60
from echopype.echodata.convention import sonarnetcdf_1
from echopype.convert.api import _check_file, SONAR_MODELS

pytestmark = pytest.mark.skip(reason="Removed Parsed2Zarr")

test_bucket_name = "echopype-test"
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


@pytest.fixture()
def s3_base():
    # writable local S3 system
    import shlex
    import subprocess

    try:
        # should fail since we didn't start server yet
        r = requests.get(endpoint_uri)
    except:  # noqa
        pass
    else:
        if r.ok:
            raise RuntimeError("moto server already up")
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    proc = subprocess.Popen(
        shlex.split("moto_server s3 -p %s" % port),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    timeout = 5
    while timeout > 0:
        try:
            print("polling for moto server")
            r = requests.get(endpoint_uri)
            if r.ok:
                break
        except:  # noqa
            pass
        timeout -= 0.1
        time.sleep(0.1)
    print("server up")
    yield
    print("moto done")
    proc.terminate()
    proc.wait()


def get_boto3_client():
    from botocore.session import Session

    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_uri)


@pytest.fixture()
def s3(s3_base):
    client = get_boto3_client()
    client.create_bucket(Bucket=test_bucket_name, ACL="public-read")

    S3FileSystem.clear_instance_cache()
    s3 = S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint_uri})
    s3.invalidate_cache()
    yield s3


@pytest.fixture
def ek60_path(test_path):
    return test_path["EK60"]


def compare_zarr_vars(
    ed_zarr: xr.Dataset, ed_no_zarr: xr.Dataset, var_to_comp: List[str], ed_path
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compares the dask variables in ``ed_zarr`` against their
    counterparts in ``ed_no_zarr`` by computing the dask results
    and using xarray to make sure the variables are identical.
    Additionally, this function will drop all of these compared
    variables.

    Parameters
    ----------
    ed_zarr : xr.Dataset
        EchoData object with variables that were written directly
        to a zarr and then loaded with dask
    ed_no_zarr : xr.Dataset
        An in-memory EchoData object
    var_to_comp : List[str]
        List representing those variables that were written directly
        to a zarr and then loaded with dask
    ed_path : str
        EchoData group (e.g. "Sonar/Beam_group1")

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        Datasets ``ed_zarr`` and ``ed_no_zarr``, respectively with
        ``var_to_comp`` removed.
    """

    for var in var_to_comp:
        for chan in ed_zarr[ed_path][var].channel:
            # here we compute to make sure values are being compared, rather than just shapes
            var_zarr = ed_zarr[ed_path][var].sel(channel=chan).compute()
            var_no_zarr = ed_no_zarr[ed_path][var].sel(channel=chan)

            assert var_zarr.identical(var_no_zarr)

    ed_zarr[ed_path] = ed_zarr[ed_path].drop_vars(var_to_comp)
    ed_no_zarr[ed_path] = ed_no_zarr[ed_path].drop_vars(var_to_comp)
    return ed_zarr, ed_no_zarr


@pytest.mark.parametrize(
    ["raw_file", "sonar_model", "destination_path"],
    [
        ("L0003-D20040909-T161906-EK60.raw", "EK60", "swap"),
        pytest.param(
            "L0003-D20040909-T161906-EK60.raw",
            "EK60",
            "no_swap",
            marks=pytest.mark.xfail(
                run=False,
                reason="Expected out of memory error. See https://github.com/OSOceanAcoustics/echopype/issues/489",
            ),
        ),
    ],
    ids=["noaa_offloaded", "noaa_not_offloaded"],
)
def test_raw2zarr(raw_file, sonar_model, destination_path, ek60_path):
    """Tests for memory expansion relief"""
    import os
    from tempfile import TemporaryDirectory
    from echopype.echodata.echodata import EchoData

    name = os.path.basename(raw_file).replace(".raw", "")
    fname = f"{name}__{destination_path}.zarr"
    file_path = ek60_path / raw_file
    echodata = open_raw(
        raw_file=file_path, sonar_model=sonar_model, destination_path=destination_path
    )
    # Most likely succeed if it doesn't crash
    assert isinstance(echodata, EchoData)
    with TemporaryDirectory() as tmpdir:
        output_save_path = os.path.join(tmpdir, fname)
        echodata.to_zarr(output_save_path)
        # If it goes all the way to here it is most likely successful
        assert os.path.exists(output_save_path)

    if echodata.parsed2zarr_obj.store is not None:
        temp_zarr_path = echodata.parsed2zarr_obj.store

        del echodata

        # make sure that the temporary zarr was deleted
        assert temp_zarr_path.fs.exists(temp_zarr_path.root) is False


@pytest.mark.parametrize(
    ["path_model", "raw_file", "sonar_model"],
    [
        ("EK60", os.path.join("ncei-wcsd", "Summer2017-D20170615-T190214.raw"), "EK60"),
        ("EK60", "DY1002_EK60-D20100318-T023008_rep_freq.raw", "EK60"),
        ("EK80", "Summer2018--D20180905-T033113.raw", "EK80"),
        ("EK80_CAL", "2018115-D20181213-T094600.raw", "EK80"),
        ("EK80", "Green2.Survey2.FM.short.slow.-D20191004-T211557.raw", "EK80"),
        ("EK80", "2019118 group2survey-D20191214-T081342.raw", "EK80"),
    ],
    ids=[
        "ek60_summer_2017",
        "ek60_rep_freq",
        "ek80_summer_2018",
        "ek80_bb_w_cal",
        "ek80_short_slow",
        "ek80_grp_2_survey",
    ],
)
def test_direct_to_zarr_integration(
    path_model: str, raw_file: str, sonar_model: str, test_path: dict
) -> None:
    """
    Integration Test that ensure writing variables
    directly to a temporary zarr store and then assigning
    them to the EchoData object create an EchoData object
    that is identical to the method of not writing directly
    to a zarr.

    Parameters
    ----------
    path_model: str
        The key in ``test_path`` pointing to the appropriate
        directory containing ``raw_file``
    raw_file: str
        The raw file to test
    sonar_model: str
        The sonar model corresponding to ``raw_file``
    test_path: dict
        A dictionary of all the model paths.

    Notes
    -----
    This test should only be conducted with small raw files
    as DataSets must be loaded into RAM!
    """

    raw_file_path = test_path[path_model] / raw_file

    ed_zarr = open_raw(raw_file_path, sonar_model=sonar_model, max_mb=100, destination_path="swap")
    ed_no_zarr = open_raw(raw_file_path, sonar_model=sonar_model)

    for grp in ed_zarr.group_paths:
        # remove conversion time so we can do a direct comparison
        if "conversion_time" in ed_zarr[grp].attrs:
            del ed_zarr[grp].attrs["conversion_time"]
            del ed_no_zarr[grp].attrs["conversion_time"]

        # Compare angle, power, complex, if zarr drop the zarr variables and compare datasets
        if grp == "Sonar/Beam_group2":
            var_to_comp = ["angle_athwartship", "angle_alongship", "backscatter_r"]
            ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)

        if grp == "Sonar/Beam_group1":
            if "backscatter_i" in ed_zarr[grp]:
                var_to_comp = ["backscatter_r", "backscatter_i"]
            else:
                var_to_comp = ["angle_athwartship", "angle_alongship", "backscatter_r"]

            ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)

        assert ed_zarr[grp] is not None

    if ed_zarr.parsed2zarr_obj.store is not None:
        temp_zarr_path = ed_zarr.parsed2zarr_obj.store

        del ed_zarr

        # make sure that the temporary zarr was deleted
        assert temp_zarr_path.fs.exists(temp_zarr_path.root) is False


class TestParsed2Zarr:
    sample_file = "L0003-D20040909-T161906-EK60.raw"
    sonar_model = "EK60"
    xml_path = None
    convert_params = None
    storage_options = {}
    max_mb = 100
    ek60_expected_shapes = {
        "angle_alongship": (9923, 3, 10417),
        "angle_athwartship": (9923, 3, 10417),
        "channel": (3,),
        "timestamp": (9923,),
        "power": (9923, 3, 10417),
    }

    @pytest.fixture(scope="class")
    def ek60_parsed2zarr_obj(self, ek60_parser_obj):
        return Parsed2ZarrEK60(ek60_parser_obj)

    @pytest.fixture(scope="class")
    def ek60_parsed2zarr_obj_w_df(self, ek60_parsed2zarr_obj):
        ek60_parsed2zarr_obj._create_zarr_info()
        ek60_parsed2zarr_obj.datagram_df = pd.DataFrame.from_dict(
            ek60_parsed2zarr_obj.parser_obj.zarr_datagrams
        )
        # convert channel column to a string
        ek60_parsed2zarr_obj.datagram_df["channel"] = ek60_parsed2zarr_obj.datagram_df["channel"].astype(str)
        yield ek60_parsed2zarr_obj

    def _get_storage_options(self, dest_path: Optional[str]) -> Optional[dict]:
        """Retrieve storage options for destination path"""
        dest_storage_options = None
        if dest_path is not None and "s3://" in dest_path:
            dest_storage_options = {"anon": False, "client_kwargs": {"endpoint_url": endpoint_uri}}
        return dest_storage_options

    @pytest.fixture(scope="class")
    def ek60_parser_obj(self, test_path):
        folder_path = test_path[self.sonar_model]
        raw_file = str(folder_path / self.sample_file)

        file_chk, xml_chk = _check_file(
            raw_file, self.sonar_model, self.xml_path, self.storage_options
        )

        if SONAR_MODELS[self.sonar_model]["xml"]:
            params = xml_chk
        else:
            params = "ALL"

        # obtain dict associated with directly writing to zarr
        dgram_zarr_vars = SONAR_MODELS[self.sonar_model]["dgram_zarr_vars"]

        # Parse raw file and organize data into groups
        parser = SONAR_MODELS[self.sonar_model]["parser"](
            file_chk,
            params=params,
            storage_options=self.storage_options,
            dgram_zarr_vars=dgram_zarr_vars,
        )

        # Parse the data
        parser.parse_raw()
        return parser

    @pytest.mark.parametrize(
        ["sonar_model", "p2z_class"],
        [
            (None, Parsed2Zarr),
            ("EK60", Parsed2ZarrEK60),
        ],
    )
    def test_constructor(self, sonar_model, p2z_class, ek60_parser_obj):
        if sonar_model is None:
            p2z = p2z_class(None)
            assert p2z.parser_obj is None
            assert p2z.temp_zarr_dir is None
            assert p2z.zarr_file_name is None
            assert p2z.store is None
            assert p2z.zarr_root is None
            assert p2z._varattrs == sonarnetcdf_1.yaml_dict["variable_and_varattributes"]
        else:
            p2z = p2z_class(ek60_parser_obj)
            assert isinstance(p2z.parser_obj, SONAR_MODELS[self.sonar_model]["parser"])
            assert p2z.sonar_model == self.sonar_model

    @pytest.mark.parametrize("dest_path", [None, "./", f"s3://{test_bucket_name}/my-dir/"])
    def test__create_zarr_info(self, ek60_parsed2zarr_obj, dest_path, s3):
        dest_storage_options = self._get_storage_options(dest_path)

        ek60_parsed2zarr_obj._create_zarr_info(dest_path, dest_storage_options=dest_storage_options)

        zarr_store = ek60_parsed2zarr_obj.store
        zarr_root = ek60_parsed2zarr_obj.zarr_root

        assert isinstance(zarr_store, FSMap)
        assert isinstance(zarr_root, ZGroup)
        assert zarr_store.root.endswith(".zarr")

        if dest_path is None:
            assert os.path.dirname(zarr_store.root) == str(DEFAULT_ZARR_TEMP_DIR)
            assert ek60_parsed2zarr_obj.temp_zarr_dir == str(DEFAULT_ZARR_TEMP_DIR)
        elif "s3://" not in dest_path:
            shutil.rmtree(zarr_store.root)

    def test__close_store(self, ek60_parsed2zarr_obj):
        ek60_parsed2zarr_obj._create_zarr_info()

        zarr_store = ek60_parsed2zarr_obj.store

        # Initially metadata should not exist
        assert not zarr_store.fs.exists(zarr_store.root + "/.zmetadata")

        # Close store will consolidate metadata
        ek60_parsed2zarr_obj._close_store()

        # Now metadata should exist
        assert zarr_store.fs.exists(zarr_store.root + "/.zmetadata")
        
    def test__write_power(self, ek60_parsed2zarr_obj_w_df):
        # There shouldn't be any group here
        assert "power" not in ek60_parsed2zarr_obj_w_df.zarr_root
        
        ek60_parsed2zarr_obj_w_df._write_power(
            df=ek60_parsed2zarr_obj_w_df.datagram_df,
            max_mb=self.max_mb
        )
        
        # There should now be power group
        assert "power" in ek60_parsed2zarr_obj_w_df.zarr_root
        
        for k, arr in ek60_parsed2zarr_obj_w_df.zarr_root["/power"].arrays():
            assert arr.shape == self.ek60_expected_shapes[k]
    
    def test__write_angle(self, ek60_parsed2zarr_obj_w_df):
        # There shouldn't be any group here
        assert "angle" not in ek60_parsed2zarr_obj_w_df.zarr_root
        
        ek60_parsed2zarr_obj_w_df._write_angle(
            df=ek60_parsed2zarr_obj_w_df.datagram_df,
            max_mb=self.max_mb
        )
        # There should now be angle group
        assert "angle" in ek60_parsed2zarr_obj_w_df.zarr_root
        
        for k, arr in ek60_parsed2zarr_obj_w_df.zarr_root["/angle"].arrays():
            assert arr.shape == self.ek60_expected_shapes[k]
    
    def test_power_dataarray(self, ek60_parsed2zarr_obj_w_df):
        power_dataarray = ek60_parsed2zarr_obj_w_df.power_dataarray
        assert isinstance(power_dataarray, xr.DataArray)
        
        assert power_dataarray.name == "backscatter_r"
        assert power_dataarray.dims == ("ping_time", "channel", "range_sample")
        assert power_dataarray.shape == self.ek60_expected_shapes["power"]
        
    def test_angle_dataarrays(self, ek60_parsed2zarr_obj_w_df):
        angle_athwartship, angle_alongship = ek60_parsed2zarr_obj_w_df.angle_dataarrays
        assert isinstance(angle_athwartship, xr.DataArray)
        assert isinstance(angle_alongship, xr.DataArray)
        
        assert angle_alongship.name == "angle_alongship"
        assert angle_alongship.dims == ("ping_time", "channel", "range_sample")
        assert angle_alongship.shape == self.ek60_expected_shapes["angle_alongship"]
        
        assert angle_athwartship.name == "angle_athwartship"
        assert angle_athwartship.dims == ("ping_time", "channel", "range_sample")
        assert angle_athwartship.shape == self.ek60_expected_shapes["angle_athwartship"]
