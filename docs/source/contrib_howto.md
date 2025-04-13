(contrib-howto)=
# How to add a functionality

Thank you for your interests in contributing to Echopype!

Check out our [](contrib-roadmap_priorities) to see what would make Echopype more powerful for analyzing echosounder data. Feel free to ping the maintainers (@leewujung, @ctuguina) for discussion and questions in existing issues, open a new issue, or create a PR directly.

:::{tip}
We encourage all code contributions to be accompanied by tests and documentations (doctrings and inline comments).
We may ask for these when reviewing the PRs.
If you have added new tests but the [GitHub Actions for continuous integration](#github-actions-for-continuous-integration-ci) need approval to run, ping the maintainers (@leewujung, @ctuguinay) to get them started.
:::



## Add rule-based processing functions

Many echosounder data processing functions (e.g., [](contrib-roadmap_algorithms)) operate on Sv data. In Echopype, Sv datasets (generated from `compute_Sv`) are standard [xarray `Dataset`](https://docs.xarray.dev/en/latest/user-guide/data-structures.html#dataset) and can be saved in Zarr or netCDF format. Therefore:

**If you have an algorithm written using xarray operations**: you can likely create a new function in the Echopype subpackage you think your function would best sit in and directly transplant your algorithm there. Just make sure that the dimension/coordinate names match between what your function need and what Echopype Sv dataset contains.

**If you have an algorithm written using numpy, scipy, or other common libraries**: we recommend that you replace the pure index-based slicing/indexing operations (e.g., `i=1`, `j=2`, ...) with xarray label-aware operations (e.g., `depth=1`, `ping_time="2025-04-12T12:00:00"`, ...). This makes the implementation much more readable and easier to debug, and have the added advantage of directly leveraging xarray's integration with numpy, dask, and zarr to allow distributed, out-of-core computing of large data.

**If you have an algorithm using image processing functions**: check out [dask-image](https://image.dask.org/en/stable/) to see if you can leverage any implementations that are already scalable.

Typically:
- A processing function would either add data variables to the Sv dataset (e.g., `consolidate.add_latlon`) or return a new xarray `Dataset` or `DataArray` (e.g., `mask.apply_mask`).
- The functions should accept input data either as an in-memory or lazy-loaded data or a path to local or remote storage locations (e.g., cloud, http server). If the input data is in netCDF or Zarr, this is easily supported by xarray. The only thing to watch out for is that, for a remote path, access credentials need to be provided via adding a `storage_options` argument.

:::{tip}
If your algorithm uses a library that is not currently an Echopype [dependency](https://github.com/OSOceanAcoustics/echopype/blob/main/requirements.txt), please add it into the `requirements.txt` file.
:::
