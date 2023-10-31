from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_datagram_max_shape(datagram_dict: Dict[Any, List[np.ndarray]]) -> Optional[Tuple[int]]:
    """
    Get the max shape across all channels for a given datagram.
    """

    def _max_shape(shapes: List[Tuple]) -> Tuple:
        """Go through each shape and grab
        the max value from each dimension"""
        max_shape = None
        for shape in shapes:
            if max_shape is None:
                max_shape = list(shape)
            for i in range(0, len(shape)):
                if shape[i] > max_shape[i]:
                    max_shape[i] = shape[i]

        return tuple(max_shape)

    # Go through each array list and get the max shape
    # then for each channel append max shape to the n number of arrays
    arr_shapes = []
    for arr_list in datagram_dict.values():
        if arr_list is not None and not all(
            (arr is None) or (arr.size == 0) for arr in arr_list
        ):  # only if there's data in the channel
            arr_shapes.append((len(arr_list),) + max(i.shape for i in arr_list))
    if len(arr_shapes) == 0:
        return None
    return _max_shape(arr_shapes)


def calc_final_shapes(
    data_types: List[str], ping_data_dict: Dict[Any, Dict[Any, List[np.ndarray]]]
) -> Dict[str, Optional[Tuple[int]]]:
    """Calculate the final shapes for each data type.
    The final shape is the max shape across all channels.

    Example output:
    ```
    {
        'power': (9923, 10417),
        'angle': (9923, 10417, 2),
        'complex': None
    }
    ```

    Parameters
    ----------
    data_types : list
        List of data types to calculate final shapes for
    ping_data_dict : dict
        Dictionary of ping data for each data type

    Returns
    -------
    dict
        Dictionary of final shapes for each data type
    """
    # Compute the max dimension shape for each data type
    datagram_max_shapes = {}
    for data_type in data_types:
        # Get the max shape across all channels for a given datagram
        max_shape = _get_datagram_max_shape(ping_data_dict[data_type])
        if max_shape:
            if data_type == "angle":
                # Angle data has 2 variables within,
                # so just take the first 2 dimension shapes
                max_shape = max_shape[:2]
            datagram_max_shapes[data_type] = max_shape

    # Compute final shapes for each data type
    data_type_shapes = {}
    for data_type in data_types:
        # Check the number of channels for a given data type
        n_channels = len(ping_data_dict[data_type])
        max_shape = datagram_max_shapes.get(data_type, None)
        if n_channels == 0 or max_shape is None:
            # If no channels, then set the shape to None
            # since this means no data for this data type
            data_type_shapes[data_type] = None
        elif data_type == "angle":
            # Add 2 to the max shape since angle is 2D array
            data_type_shapes[data_type] = datagram_max_shapes[data_type] + (2,)
        else:
            # For all other data types, just add the max shape
            data_type_shapes[data_type] = datagram_max_shapes[data_type]

    return data_type_shapes
