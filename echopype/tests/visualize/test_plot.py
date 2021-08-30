import echopype
from echopype.testing import TEST_DATA_FOLDER

import pytest
from xarray.plot.facetgrid import FacetGrid

ek60_path = TEST_DATA_FOLDER / "ek60"
ek80_path = TEST_DATA_FOLDER / "ek80"
azfp_path = TEST_DATA_FOLDER / "azfp"
ad2cp_path = TEST_DATA_FOLDER / "ad2cp"


@pytest.mark.parametrize(
    (
        "filepath",
        "sonar_model",
        "azfp_xml_path",
    ),
    [
        (
            ek60_path / "ncei-wcsd" / "Summer2017-D20170615-T190214.raw",
            "EK60",
            None,
        ),
        (ek80_path / "D20190822-T161221.raw", "EK80", None),
        pytest.param(
            ek80_path / "D20170912-T234910.raw",
            "EK80",
            None,
            marks=pytest.mark.xfail(
                reason="Has quadrant dim.. Doesn't know how to deal with that."
            ),
        ),
        (
            azfp_path / "ooi" / "17032923.01A",
            "AZFP",
            azfp_path / "ooi" / "17032922.XML",
        ),
        (
            azfp_path / "ooi" / "17032923.01A",
            "AZFP",
            azfp_path / "ooi" / "17032922.XML",
        ),
        pytest.param(
            ad2cp_path / "raw" / "090" / "rawtest.090.00001.ad2cp",
            "AD2CP",
            None,
            marks=pytest.mark.xfail(
                run=False,
                reason="Doesn't know how to deal with ADCP.. backscatter_r not available.",
            ),
        ),
    ],
)
def test_plot_multi(
    filepath,
    sonar_model,
    azfp_xml_path,
):
    # TODO: Need to figure out how to compare the actual rendered plots
    ed = echopype.open_raw(filepath, sonar_model, azfp_xml_path)
    plot = echopype.visualize.create_echogram(ed)
    assert isinstance(plot, FacetGrid) is True
