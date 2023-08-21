# Data processing

Echopype data processing funcionalities are structured into different subpackages with expandability and a series of [data processing levels](processing-levels) in mind. Once the data is converted from the raw instrument data files to standardized [`EchoData` objects](data-format:echodata-object) (or stored in `.zarr` or `.nc` format) and calibrated, the core input and output of most subsequent functions are generic [xarray `Datasets`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html). This design allows new processing functions be easily added without needing to understand specialized objects, other than functions needing access of data stored only in the raw-converted `EchoData` objects.

The section [**Data processing functionalities**](data-proc:functions) provides information for current processing functions and their usage.

The section [**Additional information for processed data**](data-proc:additional) provides on some aspects of processed data that may require additional explanation to fully understand the representation and underlying operations.
