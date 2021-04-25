from typing import Union

import xarray as xr

from ..convert.parse_azfp import ParseAZFP
from ..convert.parse_ek60 import ParseEK60
from ..convert.parse_ek80 import ParseEK80
from ..convert.set_groups_azfp import SetGroupsAZFP
from ..convert.set_groups_ek60 import SetGroupsEK60
from ..convert.set_groups_ek80 import SetGroupsEK80


# Functions to help with dask delaying
def __parse_raw(
    parser_class, file, params, storage_options
) -> Union[ParseAZFP, ParseEK60, ParseEK80]:
    """
    Delayes the parsing of raw file
    """
    parser = parser_class(
        file=file, params=params, storage_options=storage_options
    )
    parser.parse_raw()
    return parser


def __get_set_grouper(
    set_groups_class, parser, input_file, output_path, sonar_model, params
) -> Union[SetGroupsAZFP, SetGroupsEK60, SetGroupsEK80]:
    """
    Delayes the set group function of converted file
    """
    setgrouper = set_groups_class(
        parser_obj=parser,
        input_file=input_file,
        output_path=output_path,
        sonar_model=sonar_model,
        params=params,
    )
    return setgrouper


def __set_func(setgrouper, func) -> xr.Dataset:
    """
    Runs the appropriate set_* function from the setgrouper object
    """
    return getattr(setgrouper, func)()
