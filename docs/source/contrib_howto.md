(contrib-howto)=
# How to add a new function

Thank you for your interests in contributing to Echopype!

Check out the [](contrib-roadmap) to see what would make Echopype more powerful for analyzing echosounder data. Feel free to ping the maintainers (@leewujung, @ctuguina) for discussion and questions in existing issues, open a new issue, or create a PR directly.

:::{note}
We encourage all code contributions to be accompanied by tests and documentations (doctrings and inline comments).
We may ask for these when reviewing the PRs.
If you have added new tests but the [GitHub Actions for continuous integration](contrib:setup_CI) need approval to run, ping the maintainers to get them started.
:::



## Add rule-based processing functions

Many echosounder data processing functions (e.g., [](contrib-roadmap_algorithms)) operate on Sv data or raw data parsed from instrument-generated files. In Echopype:
- raw data are stored/accessed as [`EchoData` object](data-format:echodata-object) that can be opened using [`open_converted`](function:open-converted) from the saved Zarr or netCDF files.
- Sv data are stored in standard [xarray `Dataset`](https://docs.xarray.dev/en/latest/user-guide/data-structures.html#dataset) that can be opened [using `xr.open_dataset` and other related functions](https://docs.xarray.dev/en/stable/user-guide/io.html) from the saved Zarr or netCDF files.

Since both raw data (via the `EchoData` object) and Sv data are ultimately loaded into xarray datasets:
- **If your algorithm is written using xarray operations**: You can likely create a new function in the Echopype subpackage you think your function would best sit in (see [](contrib-roadmap_algorithms) for subpackage ideas), and directly transplant your algorithm there. Just make sure that the dimension/coordinate names match between what your function need and what Echopype Sv dataset contains.
- **If your algorithm is written using numpy, scipy, or other common libraries**: We recommend that you replace the pure index-based slicing/indexing operations (e.g., `i=1`, `j=2`, ...) with xarray label-aware operations (e.g., `depth=1`, `ping_time="2025-04-12T12:00:00"`, ...). This makes the implementation much more readable and easier to debug, and have the added advantage of directly leveraging xarray's integration with numpy, dask, and zarr to allow distributed, out-of-core computing of large data.
- **If your algorithm uses image processing functions**: Check out [dask-image](https://image.dask.org/en/stable/) to see if you can leverage any implementations that are already scalable when adding your new function.

Typically:
- A processing function would either add data variables to the Sv dataset (e.g., `consolidate.add_latlon`) or return a new xarray `Dataset` or `DataArray` (e.g., `mask.apply_mask`).
- The functions should accept input data either as an in-memory or lazy-loaded data or a path to local or remote storage locations (e.g., cloud, http server). If the input data is in netCDF or Zarr, this is easily supported by xarray. The only thing to watch out for is that, for a remote path, access credentials need to be provided via adding a `storage_options` argument.

:::{tip}
If your algorithm uses a library that is not currently an Echopype [dependency](https://github.com/OSOceanAcoustics/echopype/blob/main/requirements.txt), please add it into the `requirements.txt` file.
:::






## Steps to achieve scalability

Computational scalability is a core goal of Echopype. However, from experience we know that scalability can be hard to achieve on first try, as it depends on the specific operations in the function and the exact implementation, as well as the chunking of the data.

Therefore, we recommend breaking down the addition of a new function into 3 steps:
1. Add the function following the guidelines in the above section
    - Ensure the function works with datasets of reasonable sizes (e.g., 100 MB)
    - Add tests for the new function in the testing suites (under `echopype/tests`)
2. Benchmark function performance with different sizes of dataset and different chunking schemes, to determine if further optimization for scalability is needed
    - Watch out for unexpected memory expansion in the computing steps - this can happen due to implicit broadcasting or padding operations
3. Adjust the implementation to optimize theperformance if necessary

Most current Echopype processing functions are capable of leveraging lazy-loaded datasets for delayed computation, which may require additional tuning.

:::{tip}
The xarray documentation includes a nice starting guide to [parallel computing with dask](https://docs.xarray.dev/en/latest/user-guide/dask.html).
:::
