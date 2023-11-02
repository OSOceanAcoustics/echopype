# Data processing

Echopype data processing functionalities are structured into different subpackages with expandability and a series of [data processing levels](processing-levels) in mind. Once the data is converted from the raw instrument data files to standardized [`EchoData` objects](data-format:echodata-object) (or stored in `.zarr` or `.nc` format) and calibrated, the core input and output of most subsequent functions are generic [xarray `Datasets`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html). This design allows new processing functions be easily added without needing to understand specialized objects, other than functions needing access of data stored only in the raw-converted `EchoData` objects.

The section [**Data processing functionalities**](data-proc:functions) provides information for current processing functions and their usage.

The section [**Additional information for processed data**](data-proc:additional) provides on some aspects of processed data that may require additional explanation to fully understand the representation and underlying operations.

(data-proc:format)=
## Format of processed data

Once raw data (represented by the `EchoData` objects) are calibrated  (via [`compute_Sv`](echopype.calibrate.compute_Sv)), the calibrated data and the outputs of all subsequent [processing functions](data-process:functionalities) are generic [xarray Datasets](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset).
We currently do not follow any specific conventions for processed data, but we retain provenance information in the dataset, including the [data processing levels](./processing-levels.md).
However, whether and how data variables used in the processing will be stored remain to be determined.
