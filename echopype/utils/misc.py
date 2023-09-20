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
