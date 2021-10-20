from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ..core import SonarModelsHint
from ..core import SONAR_MODELS

@pytest.mark.parametrize(["sonar_model", "ext"], [
    ("AZFP", ".01A"),
    ("AZFP", ".01a"),
    ("AZFP", ".05C"),
    ("AZFP", ".12q"),

    ("EK60", ".raw"),
    ("EK60", ".RAW"),

    ("EK80", ".raw"),
    ("EK80", ".RAW"),

    ("EA640", ".raw"),
    ("EA640", ".RAW"),

    ("AD2CP", ".ad2cp"),
    ("AD2CP", ".AD2CP"),
])
def test_file_extension_validation(sonar_model: "SonarModelsHint", ext: str):
    SONAR_MODELS[sonar_model]["validate_ext"](ext)


@pytest.mark.parametrize(["sonar_model", "ext"], [
    ("AZFP", ".001A"),
    ("AZFP", ".01AA"),
    ("AZFP", ".01aa"),
    ("AZFP", ".05AA"),
    ("AZFP", ".07!"),
    ("AZFP", ".01!"),
    ("AZFP", ".0!A"),
    ("AZFP", ".012"),
    ("AZFP", ".0AA"),
    ("AZFP", ".AAA"),
    ("AZFP", "01A"),

    ("EK60", "raw"),
    ("EK60", ".foo"),

    ("EK80", "raw"),
    ("EK80", ".foo"),

    ("EA640", "raw"),
    ("EA640", ".foo"),

    ("AD2CP", "ad2cp"),
    ("AD2CP", ".foo"),
])
def test_file_extension_validation_should_fail(sonar_model: "SonarModelsHint", ext: str):
    try:
        SONAR_MODELS[sonar_model]["validate_ext"](ext)
    except ValueError:
        pass
    else:
        raise ValueError(f"\"{ext}\" should have been rejected for sonar model {sonar_model}")
