from pathlib import Path
from echopype.convert import Convert


def test_2in1_ek80_conversion():
    file = Path("./echopype/test_data/ek80/Green2.Survey2.FM.short.slow.-D20191004-T211557.raw").resolve()
    nc_path = file.parent.joinpath(file.stem+".nc")
    tmp = Convert(str(file), model="EK80")
    tmp.raw2nc()
    del tmp
    nc_path.unlink()
