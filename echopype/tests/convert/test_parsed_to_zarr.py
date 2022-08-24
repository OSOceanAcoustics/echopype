import pytest
from echopype import open_raw


@pytest.fixture
def ek60_path(test_path):
    print(test_path)
    return test_path['EK60']


def compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, ed_path):
    # TODO: add docstring and comments

    for var in var_to_comp:

        for chan in ed_zarr[ed_path][var].channel:

            # here we compute to make sure values are being compared, rather than just shapes
            var_zarr = ed_zarr[ed_path][var].sel(channel=chan).compute()
            var_no_zarr = ed_no_zarr[ed_path][var].sel(channel=chan)

            assert var_zarr.identical(var_no_zarr)

    ed_zarr[ed_path] = ed_zarr[ed_path].drop(var_to_comp)
    ed_no_zarr[ed_path] = ed_no_zarr[ed_path].drop(var_to_comp)
    return ed_zarr, ed_no_zarr


@pytest.mark.parametrize(
    ["raw_file", "sonar_model", "offload_to_zarr"],
    [
        ("L0003-D20040909-T161906-EK60.raw", "EK60", True),
        pytest.param(
            "L0003-D20040909-T161906-EK60.raw",
            "EK60",
            False,
            marks=pytest.mark.xfail(
                run=False,
                reason="Expected out of memory error. See https://github.com/OSOceanAcoustics/echopype/issues/489",
            ),
        ),
    ],
    ids=["noaa_offloaded", "noaa_not_offloaded"],
)
def test_raw2zarr(raw_file, sonar_model, offload_to_zarr, ek60_path):
    """Tests for memory expansion relief"""
    import os
    from tempfile import TemporaryDirectory
    from echopype.echodata.echodata import EchoData
    name = os.path.basename(raw_file).replace('.raw', '')
    fname = f"{name}__{offload_to_zarr}.zarr"
    file_path = ek60_path / raw_file
    echodata = open_raw(
        raw_file=file_path,
        sonar_model=sonar_model,
        offload_to_zarr=offload_to_zarr
    )
    # Most likely succeed if it doesn't crash
    assert isinstance(echodata, EchoData)
    with TemporaryDirectory() as tmpdir:
        output_save_path = tmpdir + f"/{fname}"
        echodata.to_zarr(output_save_path)
        # If it goes all the way to here it is most likely successful
        assert os.path.exists(output_save_path)


@pytest.mark.parametrize(
    ["path_model", "raw_file", "sonar_model"],
    [
        ("EK60", "ncei-wcsd/Summer2017-D20170615-T190214.raw", "EK60"),
        ("EK60", "DY1002_EK60-D20100318-T023008_rep_freq.raw", "EK60"),
        ("EK80",  "Summer2018--D20180905-T033113.raw", "EK80"),
        # ("EK80_CAL" / "", "EK80"),
    ],
    ids=["ek60_summer_2017", "ek60_rep_freq", "ek80_summer_2018"],
)
#
# converted_raw_paths_v05x = [
#                                 ek80_path / "ek80-Summer2018--D20180905-T033113-ep-v05x.nc",
#                                 ek80_path / "ek80-2018115-D20181213-T094600-ep-v05x.nc",
#                                 ek80_path / "ek80-2019118-group2survey-D20191214-T081342-ep-v05x.nc",
#                                 ek80_path / "ek80-Green2-Survey2-FM-short-slow-D20191004-T211557-ep-v05x.nc"
def test_writing_directly_to_zarr(path_model, raw_file, sonar_model, test_path):
    """
    Tests that ensure writing variables directly to a
    temporary zarr store and then assigning them to
    the EchoData object create an EchoData object that
    is identical to the method of not writing directly
    to a zarr. This test should only be conducted with
    small raw files as DataSets must be loaded into RAM.
    """

    raw_file_path = test_path[path_model] / raw_file

    ed_zarr = open_raw(raw_file_path, sonar_model=sonar_model, offload_to_zarr=True, max_zarr_mb=100)
    ed_no_zarr = open_raw(raw_file_path, sonar_model=sonar_model, offload_to_zarr=False)

    for grp in ed_zarr.group_paths:

        if "conversion_time" in ed_zarr[grp].attrs:
            del ed_zarr[grp].attrs["conversion_time"]
            del ed_no_zarr[grp].attrs["conversion_time"]

        # Compare angle, power, complex, if zarr drop the zarr variables and compare datasets
        if grp == "Sonar/Beam_group2":
            var_to_comp = ['angle_athwartship', 'angle_alongship', 'backscatter_r']
            ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)

        if grp == "Sonar/Beam_group1":

            if 'backscatter_i' in ed_zarr[grp]:
                var_to_comp = ['backscatter_r', 'backscatter_i']
            else:
                var_to_comp = ['angle_athwartship', 'angle_alongship', 'backscatter_r']

            ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)

        assert ed_zarr[grp].identical(ed_no_zarr[grp])
