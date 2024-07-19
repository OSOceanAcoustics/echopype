import re
from typing import List, Optional, Union

import xarray as xr


def _parse_freq_diff_eq(
    freqABEq: Optional[str] = None,
    chanABEq: Optional[str] = None,
) -> List[Union[List[float], List[str], str, Union[float, int]]]:
    """
    Checks if either `freqABEq` or `chanABEq` is provided and parse the arguments accordingly
    from the frequency diffrencing criteria.

    Parameters
    ----------
    freqABEq : str, optional
        The equation for frequency-differencing using frequency values.
    chanABEq : str, optional
        The equation for frequency-differencing using channel names.

    Returns
    -------
    List[Union[List[float], List[str], str, Union[float, int]]]
        A list containing the parsed arguments for frequency-differencing, where the first element
        corresponds to `freqAB`, the second element corresponds to `chanAB`, the third element
        corresponds to `operator`, the fourth element corresponds to `diff`.

    Raises
    ------
    ValueError
        If `operator` is not a valid operator.
        If both `freqABEq` and `chanABEq` are provided.
        If neither `freqABEq` nor `chanABEq` is provided.
        If `freqAB` or `chanAB` is not a list of length 2 with unique elements.
    TypeError
        If `diff` is not a float or an int.
        If `freqABEq` or `chanABEq` is not a valid equation.
    """

    if (freqABEq is None) and (chanABEq is None):
        raise ValueError("Either freqAB or chanAB must be given!")
    elif (freqABEq is not None) and (chanABEq is not None):
        raise ValueError("Only one of freqAB or chanAB should be given, but not both!")
    elif freqABEq is not None:
        freqAPattern = r"(?P<freqA>\d*\.?\d+)\s*(?P<unitA>\w?)Hz"
        freqBPattern = r"(?P<freqB>\d*\.?\d+)\s*(?P<unitB>\w?)Hz"
        operatorPattern = r"\s*(?P<cmp>\S*?)\s*"
        rhsPattern = r"(?P<db>\d*\.?\d+)\s*dB"
        diffMatcher = re.compile(
            freqAPattern + r"\s*-\s*" + freqBPattern + operatorPattern + rhsPattern
        )
        eqMatched = diffMatcher.match(freqABEq)
        if eqMatched is None:
            raise TypeError("Invalid freqAB Equation!")
        operator = eqMatched["cmp"]
        if operator not in [">", "<", "<=", ">=", "=="]:
            raise ValueError("Invalid operator!")
        freqMultiplier = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9}
        freqA = float(eqMatched["freqA"]) * freqMultiplier[eqMatched["unitA"]]
        freqB = float(eqMatched["freqB"]) * freqMultiplier[eqMatched["unitB"]]
        freqAB = [freqA, freqB]
        if len(set(freqAB)) != 2:
            raise ValueError("freqAB must be a list of length 2 with unique elements!")
        diff = float(eqMatched["db"])
        return [freqAB, None, operator, diff]
    elif chanABEq is not None:
        chanAPattern = r"(?P<chanA>\".+\")\s*"
        chanBPattern = r"(?P<chanB>\".+\")\s*"
        operatorPattern = r"\s*(?P<cmp>\S*?)\s*"
        rhsPattern = r"(?P<db>\d*\.?\d+)\s*dB"
        diffMatcher = re.compile(
            chanAPattern + r"\s*-\s*" + chanBPattern + operatorPattern + rhsPattern
        )
        eqMatched = diffMatcher.match(chanABEq)
        if eqMatched is None:
            raise TypeError("Invalid chanAB Equation!")
        operator = eqMatched["cmp"]
        if operator not in [">", "<", "<=", ">=", "=="]:
            raise ValueError("Invalid operator!")
        chanAB = [eqMatched["chanA"][1:-1], eqMatched["chanB"][1:-1]]
        if len(set(chanAB)) != 2:
            raise ValueError("chanAB must be a list of length 2 with unique elements!")
        diff = float(eqMatched["db"])
        return [None, chanAB, operator, diff]


def _check_freq_diff_source_Sv(
    source_Sv: xr.Dataset,
    freqAB: Optional[List[float]] = None,
    chanAB: Optional[List[str]] = None,
) -> None:
    """
    Ensures that ``source_Sv`` contains ``channel`` as a coordinate and
    ``frequency_nominal`` as a variable, the provided list input
    (``freqAB`` or ``chanAB``) are contained in the coordinate ``channel``
    or variable ``frequency_nominal``, and ``source_Sv`` does not have
    repeated values for ``channel`` and ``frequency_nominal``.

    Parameters
    ----------
    source_Sv: xr.Dataset
        A Dataset that contains the Sv data to create a mask for
    freqAB: list of float, optional
        The pair of nominal frequencies to be used for frequency-differencing, where
        the first element corresponds to ``freqA`` and the second element corresponds
        to ``freqB``
    chanAB: list of float, optional
        The pair of channels that will be used to select the nominal frequencies to be
        used for frequency-differencing, where the first element corresponds to ``freqA``
        and the second element corresponds to ``freqB``
    """

    # check that channel and frequency nominal are in source_Sv
    if "channel" not in source_Sv.coords:
        raise ValueError("The Dataset defined by source_Sv must have channel as a coordinate!")
    elif "frequency_nominal" not in source_Sv.variables:
        raise ValueError(
            "The Dataset defined by source_Sv must have frequency_nominal as a variable!"
        )

    # make sure that the channel values are not repeated in source_Sv and
    # elements of chanAB are in channel
    if chanAB is not None:
        if len(set(source_Sv.channel.values)) < source_Sv.channel.size:
            raise ValueError(
                "The provided source_Sv contains repeated channel values, this is not allowed!"
            )
        if not all([chan in source_Sv.channel for chan in chanAB]):
            raise ValueError(
                "The provided list input chanAB contains values that are "
                "not in the channel coordinate!"
            )

    # make sure that the frequency_nominal values are not repeated in source_Sv and
    # elements of freqAB are in frequency_nominal
    if freqAB is not None:
        if len(set(source_Sv.frequency_nominal.values)) < source_Sv.frequency_nominal.size:
            raise ValueError(
                "The provided source_Sv contains repeated "
                "frequency_nominal values, this is not allowed!"
            )

        if not all([freq in source_Sv.frequency_nominal for freq in freqAB]):
            raise ValueError(
                "The provided list input freqAB contains values that "
                "are not in the frequency_nominal variable!"
            )
