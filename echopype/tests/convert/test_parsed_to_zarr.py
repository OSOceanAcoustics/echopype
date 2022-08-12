import pytest
from echopype import open_raw


@pytest.fixture
def ek60_path(test_path):
    return test_path['EK60']


@pytest.fixture
def ek80_path(test_path):
    return test_path['EK80']


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


@pytest.mark.skip(reason="Full testing of writing variables directly to a zarr store has not been implemented yet.")
def test_writing_directly_to_zarr(ek60_path, ek80_path):
    """
    Tests that ensure writing variables directly to a
    temporary zarr store and then assigning them to
    the EchoData object create an EchoData object that
    is identical to the method of not writing directly
    to a zarr.
    """

    pass

    # TODO: use the below structure to compare small files
    # TODO: also create a test that runs L0003-D20040909-T161906-EK60.raw (the 95MB file that explodes)

    # ed_zarr = ep.open_raw(path_to_raw, sonar_model=sonar_model, offload_to_zarr=True, max_zarr_mb=100)
    # ed_no_zarr = ep.open_raw(path_to_raw, sonar_model=sonar_model, offload_to_zarr=False)
    #
    # for grp in ed_zarr.group_paths:
    #
    #     if "conversion_time" in ed_zarr[grp].attrs:
    #         del ed_zarr[grp].attrs["conversion_time"]
    #         del ed_no_zarr[grp].attrs["conversion_time"]
    #
    #     # Compare straight up angle, power, complex, if zarr
    #     # drop the zarr variables and compare datasets
    #
    #     if grp == "Sonar/Beam_group2":
    #         var_to_comp = ['angle_athwartship', 'angle_alongship', 'backscatter_r']
    #         ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)
    #
    #     if grp == "Sonar/Beam_group1":
    #
    #         if 'backscatter_i' in ed_zarr[grp]:
    #             var_to_comp = ['backscatter_r', 'backscatter_i']
    #         else:
    #             var_to_comp = ['angle_athwartship', 'angle_alongship', 'backscatter_r']
    #
    #         ed_zarr, ed_no_zarr = compare_zarr_vars(ed_zarr, ed_no_zarr, var_to_comp, grp)
    #
    #     assert ed_zarr[grp].identical(ed_no_zarr[grp])
