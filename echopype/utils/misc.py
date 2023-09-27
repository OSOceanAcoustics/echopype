def camelcase2snakecase(camel_case_str):
    """
    Convert string from CamelCase to snake_case
    e.g. CamelCase becomes camel_case.
    """
    idx = list(reversed([i for i, c in enumerate(camel_case_str) if c.isupper()]))
    param_len = len(camel_case_str)
    for i in idx:
        #  check if we should insert an underscore
        if i > 0 and i < param_len:
            camel_case_str = camel_case_str[:i] + "_" + camel_case_str[i:]

    return camel_case_str.lower()


def frequency_nominal_to_channel(source_Sv, frequency_nominal: int):
    """
    Given a value for a nominal frequency, returns the channel associated with it
    """
    channels = source_Sv["frequency_nominal"].coords["channel"].values
    freqs = source_Sv["frequency_nominal"].values
    chan = channels[freqs == frequency_nominal]
    assert len(chan) == 1, "Frequency not uniquely identified"
    channel = chan[0]
    return channel
