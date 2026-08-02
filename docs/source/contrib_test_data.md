(contrib:test-data)=
# Managing test data

echopype uses a collection of test data bundles to support unit and integration tests across different sonar models and file formats. These bundles are distributed through GitHub release assets and downloaded automatically using Pooch during testing.

The test data inventory is maintained in:

```text
docs/source/test_data_inventory.yml
```

This inventory serves as the central metadata registry for the test data bundles currently used by the test suite. It includes, when available:

* instrument type,
* file descriptions,
* data source,
* contributor,
* references,
* additional notes.

The inventory is validated in CI against the list of test data bundles declared in the test suite (`echopype/tests/conftest.py`). When a new test data bundle is added, the inventory should be updated accordingly.

## Inventory fields

| Field                 | Description                                                                     |
| :-------------------- | :------------------------------------------------------------------------------ |
| `documented_checksum` | SHA256 checksum of the bundle version for which the metadata has been reviewed. |
| `instrument`          | Sonar or instrument type, when known.                                           |
| `description`         | Short description of the bundle or file.                                        |
| `source`              | Origin of the dataset, when known.                                              |
| `contributor`         | Contributor or data provider, when known.                                       |
| `references`          | Related publications, documentation, or external resources.                     |
| `notes`               | Additional information or context.                                              |
| `files`               | Metadata for individual files contained in the bundle.                          |

## Adding a new test data bundle

When adding a new test data bundle:

1. Add the bundle to the Pooch registry in `echopype/tests/conftest.py`.
2. Add a corresponding entry to `docs/source/test_data_inventory.yml`.
3. Populate the available metadata fields as completely as possible.
4. If the bundle metadata has been reviewed, update `documented_checksum` to match the bundle checksum in the Pooch registry.
