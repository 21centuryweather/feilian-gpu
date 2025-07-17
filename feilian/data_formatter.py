import math

import numpy as np
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DEFAULT_FORMAT_SHAPE = 1280


def _find_formatted_shape(fmt_shape):
    """
    Determines the formatted shape based on the input.

    Args:
        fmt_shape (int, tuple, or list): The input shape which can be an integer, 
                                         a tuple, or a list. If it is an integer, 
                                         it is assumed to be both dimensions of the shape. 
                                         If it is a tuple or list, it can either have one 
                                         element (assumed to be both dimensions) or two 
                                         elements (representing the two dimensions).

    Returns:
        tuple: A tuple containing two integers representing the formatted shape.

    Notes:
        If the input is not an integer, tuple, or list, the function returns a default 
        shape defined by DEFAULT_FORMAT_SHAPE.
    """
    if isinstance(fmt_shape, int):
        return fmt_shape, fmt_shape
    if isinstance(fmt_shape, (tuple, list)):
        if len(fmt_shape) == 1:
            return fmt_shape[0], fmt_shape[0]
        return fmt_shape[0], fmt_shape[1]
    return DEFAULT_FORMAT_SHAPE, DEFAULT_FORMAT_SHAPE


def _determine_shape_before_rotation(alpha, fmt_shape):
    """
    Determine the shape of an object before rotation.

    This function calculates the shape of an object before it is rotated by a given angle.
    If the calculated shape has any non-positive dimensions, it recalculates the shape using
    the maximum dimension of the original shape.

    Parameters:
    alpha (float): The rotation angle in degrees.
    fmt_shape (tuple or list of two floats): The original shape dimensions.

    Returns:
    numpy.ndarray: The shape dimensions before rotation.
    """
    ralpha = np.deg2rad(alpha)
    r = np.array([[np.cos(ralpha), np.sin(ralpha)], [np.sin(ralpha), np.cos(ralpha)]])
    tshape = np.linalg.solve(r, np.array(fmt_shape))
    if np.any(tshape <= 0):
        max_shape = np.max(fmt_shape)
        tshape = np.linalg.solve(r, np.array([max_shape, max_shape]))
    return tshape


def _is_point_below_line(point, line):
    """
    Determines if a given point is below or on a specified line.

    Args:
        point (tuple): A tuple representing the coordinates of the point (x, y).
        line (tuple): A tuple containing two tuples, each representing the coordinates of the endpoints of the line ((x1, y1), (x2, y2)).

    Returns:
        bool: True if the point is below or on the line, False otherwise.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    return k * point[0] + b >= point[1]


def _compute_expanded_shape_and_rotated_blocks(data, alpha, fmt_shape):
    """
    Computes the expanded shape and the number of rotated blocks required to fit the given data.

    Parameters:
    data (numpy.ndarray): The input data array.
    alpha (float): The rotation angle in degrees.
    fmt_shape (tuple): The shape of the format (fm, fn).

    Returns:
    tuple: A tuple containing:
        - expanded_shape (tuple): The expanded shape of the data after rotation.
        - rotated_blocks (tuple): The number of blocks in the expanded shape.

    The following is the meaning of the prefix:
        - d: data
        - f: formatted
        - t: target
        - n: number
        - s: start
        - a: actual
        - ef: expanded formatted
    """
    dm, dn = np.shape(data)
    fm, fn = fmt_shape
    if alpha == 0:
        nm, nn = np.int32(np.ceil(dm / fm)), np.int32(np.ceil(dn / fn))
        return (nm * fm, nn * fn), (nm, nn)

    tm, tn = _determine_shape_before_rotation(alpha, fmt_shape)
    nm, nn = math.ceil(tm / dm), math.ceil(tn / dn)
    ralpha = np.deg2rad(alpha)
    r = np.array([[np.cos(ralpha), np.sin(ralpha)], [np.sin(ralpha), np.cos(ralpha)]])
    rr = np.array([[np.cos(-ralpha), -np.sin(-ralpha)], [np.sin(-ralpha), np.cos(-ralpha)]])
    dpoint1, dpoint2 = [-dn // 2, dm // 2], [dn - dn // 2, dm // 2]
    while True:
        am, an = np.int32(np.round(np.dot(r, np.array([nm * dm, nn * dn]))))
        efm, efn = am // fm * fm, an // fn * fm
        x1, x2, y1, y2 = efn // 2, efn - efn // 2, efm // 2, efm - efm // 2
        points = np.array([[-x1, -x1, x2], [-y2, y1, y1]])
        rpoints = np.dot(rr, points)
        leftok, rightok = _is_point_below_line(dpoint1, rpoints[:, 0:2]), _is_point_below_line(dpoint2, rpoints[:, 1:3])
        if leftok and rightok:
            break
        if not leftok:
            nn += 1
        if not rightok:
            nm += 1

    am, an = np.dot(r, np.array([nm * dm, nn * dn]))
    return (nm * dm, nn * dn), (np.int32(am // fm), np.int32(an // fn))


def _init_all_metrics():
    """
    Initializes a dictionary to store various evaluation metrics.

    Returns:
        dict: A dictionary with keys for different metrics, each initialized to an empty list.
              The keys include:
              - "mae": Mean Absolute Error
              - "rmse": Root Mean Squared Error
              - "r2_score": R-squared Score
              - "rel_l2_error": Relative L2 Error
              - "true_mean": True Mean
              - "pred_mean": Predicted Mean
              - "abs_mean_diff": Absolute Mean Difference
              - "rel_mean_diff": Relative Mean Difference
              - "true_std": True Standard Deviation
              - "pred_std": Predicted Standard Deviation
              - "abs_std_diff": Absolute Standard Deviation Difference
              - "rel_std_diff": Relative Standard Deviation Difference
    """
    all_metrics = {"mae": [], "rmse": [], "r2_score": [], "rel_l2_error": [],
                   "true_mean": [], "pred_mean": [], "abs_mean_diff": [], "rel_mean_diff": [],
                   "true_std": [], "pred_std": [],  "abs_std_diff": [], "rel_std_diff": []}
    return all_metrics


def _add_more_metrics(all_metrics, y_true, y_pred):
    """
    Adds various metrics to the provided dictionary `all_metrics` based on the true and predicted values.

    Parameters:
    all_metrics (dict): Dictionary to store the computed metrics.
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Metrics added:
    - rel_l2_error: Relative L2 norm error between y_true and y_pred.
    - true_mean: Mean of the true values.
    - pred_mean: Mean of the predicted values.
    - abs_mean_diff: Absolute difference between the means of y_true and y_pred.
    - rel_mean_diff: Relative difference between the means of y_true and y_pred.
    - abs_std_diff: Absolute difference between the standard deviations of y_true and y_pred.
    - rel_std_diff: Relative difference between the standard deviations of y_true and y_pred.
    - true_std: Standard deviation of the true values.
    - pred_std: Standard deviation of the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    all_metrics["rel_l2_error"].append(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))
    y_true_mean, y_pred_mean = np.mean(y_true), np.mean(y_pred)
    abs_mean_diff = np.abs(y_true_mean - y_pred_mean)
    rel_mean_diff = abs_mean_diff / y_true_mean
    all_metrics["true_mean"].append(y_true_mean)
    all_metrics["pred_mean"].append(y_pred_mean)
    all_metrics["abs_mean_diff"].append(abs_mean_diff)
    all_metrics["rel_mean_diff"].append(rel_mean_diff)
    y_true_std, y_pred_std = np.std(y_true), np.std(y_pred)
    abs_std_diff = np.abs(y_true_std - y_pred_std)
    rel_std_diff = abs_std_diff / y_true_std
    all_metrics["abs_std_diff"].append(abs_std_diff)
    all_metrics["rel_std_diff"].append(rel_std_diff)
    all_metrics["true_std"].append(y_true_std)
    all_metrics["pred_std"].append(y_pred_std)


def _compute_then_add_metrics(all_metrics, y_true, y_pred):
    """
    Compute and add various regression metrics to the provided dictionary.

    This function calculates the R^2 score, mean absolute error (MAE), and root mean squared error (RMSE)
    for the given true and predicted values, and appends these metrics to the corresponding lists in the
    `all_metrics` dictionary. It also calls an additional function to add more metrics.

    Parameters:
    all_metrics (dict): A dictionary where keys are metric names and values are lists to which the computed
                        metrics will be appended.
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    dict: The updated `all_metrics` dictionary with the new metrics appended.
    """
    all_metrics["r2_score"].append(r2_score(y_true, y_pred))
    all_metrics["mae"].append(mean_absolute_error(y_true, y_pred))
    all_metrics["rmse"].append(mean_squared_error(y_true, y_pred, squared=False))
    _add_more_metrics(all_metrics, y_true, y_pred)

    return all_metrics


class DataFormatter:
    def __init__(self, raw_data, wind_angles=None, formatted_shape=DEFAULT_FORMAT_SHAPE):
        """
        Initialize a new instance of DataFormatter.

        :param raw_data: The wind speed and topo data in one 2D matrix. The negative entries correspond to topo.
        :param wind_angles: The corresponding wind angles of the data, in clockwise direction.
        :param formatted_shape: The desired x-y shape of the network input.
        """
        self.raw_data = raw_data
        self.wind_angles = wind_angles
        self.fmt_shape = _find_formatted_shape(formatted_shape)

        self._fmt_input_data, self._fmt_output_data = None, None
        self._slice_indices = [range(1) for _ in raw_data]
        self._expanded_shapes = [(0, 0) for _ in raw_data]
        self._rotated_blocks = [(0, 0) for _ in raw_data]
        self._actual_angles = [0 for _ in raw_data]
        self._data_type = None

        self._process_raw_data_by_wind_angles(raw_data, wind_angles)
        self._init_fmt_input_and_output_data()
        self._fill_fmt_input_and_output_data()

    def get_formatted_input_data(self):
        return self._fmt_input_data

    def get_formatted_output_data(self):
        return self._fmt_output_data

    def restore_raw_output_data(self, formatted_output, raw_data_indices=None, fill_value=0):
        """
        Restore the raw output data from the formatted output data.

        Parameters:
        formatted_output (numpy.ndarray): The formatted output data to be restored.
        raw_data_indices (list of int, optional): Indices of the raw data to be restored. If None, all raw data will be restored.
        fill_value (int, optional): The value to fill in the missing data. Default is 0.

        Returns:
        list: A list of restored raw output data.

        Raises:
        AssertionError: If the shape of the formatted output data does not match the expected shape.
        """
        if not raw_data_indices:
            assert np.shape(self._fmt_input_data) == np.shape(formatted_output), 'data shape must match'

            raw_output = []
            for raw_idx in range(len(self.raw_data)):
                slice_start_idx = self._slice_indices[raw_idx].start
                single_output = self._restore_single_raw_output_data_from_slices(raw_idx, formatted_output,
                                                                                 slice_start_idx, fill_value)
                raw_output.append(single_output)
            return raw_output

        nslices = 0
        for idx in raw_data_indices:
            nslices += len(self._slice_indices[idx])
        fm, fn = self.fmt_shape
        assert (nslices, 1, fm, fn) == np.shape(formatted_output), 'data shape must match'

        raw_output = []
        slice_start_idx = 0
        for raw_idx in raw_data_indices:
            single_output = self._restore_single_raw_output_data_from_slices(raw_idx, formatted_output,
                                                                             slice_start_idx, fill_value)
            raw_output.append(single_output)
            slice_start_idx += len(self._slice_indices[raw_idx])
        return raw_output

    def compute_all_metrics(self, formatted_output, raw_data_indices=None):
        """
        Computes all metrics for the given formatted output and raw data indices.

        Args:
            formatted_output (list or array-like): The formatted output data that needs to be evaluated.
            raw_data_indices (list or array-like, optional): Indices of the raw data to be used for evaluation. 
            If None, all raw data will be used.

        Returns:
            dict: A dictionary containing all computed metrics.

        Notes:
            - The method restores the raw output data from the formatted output.
            - If raw_data_indices is not provided, it defaults to using the entire raw data.
            - Metrics are computed for each pair of truth and predicted values, and then aggregated.
        """
        predicted_output = self.restore_raw_output_data(formatted_output, raw_data_indices)
        if not raw_data_indices:
            raw_data_indices = range(len(self.raw_data))

        truth_all, pred_all = [], []
        all_metrics = _init_all_metrics()
        for (pidx, tidx) in enumerate(raw_data_indices):
            num_idx = self.raw_data[tidx] >= 0
            truth, pred = self.raw_data[tidx][num_idx], predicted_output[pidx][num_idx]
            _compute_then_add_metrics(all_metrics, truth, pred)
            pred_all.extend(pred)
            truth_all.extend(truth)
        _compute_then_add_metrics(all_metrics, truth_all, pred_all)
        return all_metrics

    def split_train_test_data(self, train_ratio, seed=None):
        """
        Splits the raw data into training and testing datasets based on the given train_ratio.

        Parameters:
        train_ratio (float): The ratio of the data to be used for training. Should be between 0 and 1.
        seed (int, optional): A seed for the random number generator to ensure reproducibility. Default is None.

        Returns:
        tuple: A tuple containing:
            - x_train (numpy.ndarray): The formatted input training data.
            - y_train (numpy.ndarray): The formatted output training data.
            - raw_train_idx (list): The indices of the raw training data.
            - x_test (numpy.ndarray): The formatted input testing data.
            - y_test (numpy.ndarray): The formatted output testing data.
            - raw_test_idx (list): The indices of the raw testing data.
        """
        split_idx = round(train_ratio * len(self.raw_data))
        if seed:
            np.random.seed(seed)
        indices = [i for i in range(len(self.raw_data))]
        np.random.shuffle(indices)
        raw_train_idx, raw_test_idx = sorted(indices[:split_idx]), sorted(indices[split_idx:])
        fmt_train_idx, fmt_test_idx = [], []
        for raw_idx in raw_train_idx:
            fmt_train_idx.extend(self._slice_indices[raw_idx])
        for raw_idx in raw_test_idx:
            fmt_test_idx.extend(self._slice_indices[raw_idx])
        x_train, y_train = self._fmt_input_data[fmt_train_idx, :, :, :], self._fmt_output_data[fmt_train_idx, :, :, :]
        x_test, y_test = self._fmt_input_data[fmt_test_idx, :, :, :], self._fmt_output_data[fmt_test_idx, :, :, :]
        return x_train, y_train, raw_train_idx, x_test, y_test, raw_test_idx

    def _process_raw_data_by_wind_angles(self, raw_data, wind_angles):
        """
        Processes raw data by rotating images based on provided wind angles.

        Parameters:
        raw_data (list): List of raw data images.
        wind_angles (list or None): List of wind angles corresponding to each image in raw_data. 
                                    If None, all wind angles are set to 0.

        Raises:
        AssertionError: If the length of raw_data and wind_angles do not match.
        AssertionError: If any wind angle is not in the range [0, 360).

        Notes:
        - If wind_angles is None, all images in raw_data are used as is, and wind_angles are set to 0.
        - If wind_angles are provided, each image in raw_data is rotated based on the corresponding wind angle.
        - The rotation is performed in multiples of 90 degrees.
        - The actual angle after rotation is stored in self._actual_angles.
        """
        if wind_angles is None:
            self.raw_data = raw_data
            self.wind_angles = [0 for _ in range(len(raw_data))]
        else:
            assert len(raw_data) == len(wind_angles), 'data and wind angles should have the same length'
            assert all([0 <= wa < 360 for wa in wind_angles]), 'wind angle should be in [0, 360)'

            self.raw_data = []
            self.wind_angles = wind_angles
            for (i, wd) in enumerate(wind_angles):
                if wd == 0:
                    self.raw_data.append(raw_data[i])
                else:
                    rotated_image, k = raw_data[i], int(wd // 90)
                    if k > 0:
                        rotated_image = np.rot90(raw_data[i], k=k)
                    self.raw_data.append(rotated_image)
                    self._actual_angles[i] = wd - k * 90

    def _init_fmt_input_and_output_data(self):
        """
        Initializes formatted input and output data by processing raw data.

        This method performs the following steps:
        1. Iterates over the raw data and corresponding angles.
        2. Computes the expanded shape and rotated blocks for each data item.
        3. Stores the expanded shapes and rotated blocks.
        4. Calculates the slice indices for each data item.
        5. Updates the total number of slices.
        6. Initializes empty arrays for formatted input and output data with the appropriate shape and data type.

        Attributes:
            total_slices (int): Total number of slices across all data items.
            fm (int): First dimension of the formatted shape.
            fn (int): Second dimension of the formatted shape.
            didx (int): Index of the current data item.
            data (ndarray): Current raw data item.
            alpha (float): Actual angle corresponding to the current data item.
            exp_shape (tuple): Expanded shape of the current data item.
            blocks (tuple): Rotated blocks of the current data item.
            curr_slices (int): Number of slices in the current data item.
            slice_shape (tuple): Shape of the slice array.
            _data_type (dtype): Data type of the raw data.
            _fmt_input_data (ndarray): Formatted input data array.
            _fmt_output_data (ndarray): Formatted output data array.
        """
        total_slices = 0
        fm, fn = self.fmt_shape
        for (didx, data) in enumerate(self.raw_data):
            alpha = self._actual_angles[didx]
            exp_shape, blocks = _compute_expanded_shape_and_rotated_blocks(data, alpha, self.fmt_shape)
            self._expanded_shapes[didx] = exp_shape
            self._rotated_blocks[didx] = blocks
            curr_slices = np.prod(blocks)
            self._slice_indices[didx] = range(total_slices, total_slices + curr_slices)
            total_slices += curr_slices
        slice_shape = (total_slices, 1, fm, fn)
        self._data_type = self.raw_data[0].dtype
        self._fmt_input_data = np.empty(shape=slice_shape, dtype=self._data_type)
        self._fmt_output_data = np.empty(shape=slice_shape, dtype=self._data_type)

    def _fill_fmt_input_and_output_data(self):
        """
        Fills the formatted input and output data arrays by expanding, padding, and processing the raw data.

        This method processes each dataset in `self.raw_data` by:
        1. Padding the data to match the expanded shapes specified in `self._expanded_shapes`.
        2. Rotating and cropping the expanded data.
        3. Creating an expanded input array with negative values from the expanded output.
        4. Splitting the expanded data into smaller slices and storing them in `self._fmt_input_data` and `self._fmt_output_data`.

        The method uses the following attributes:
        - `self.raw_data`: List of raw data arrays to be processed.
        - `self._expanded_shapes`: List of tuples specifying the target shapes for each dataset.
        - `self._rotate_then_crop_expanded_data`: Method to rotate and crop the expanded data.
        - `self._data_type`: Data type for the expanded input array.
        - `self._slice_indices`: List of slice indices for storing the formatted data.
        - `self.fmt_shape`: Tuple specifying the shape of the formatted data slices.
        - `self._fmt_input_data`: Array to store the formatted input data.
        - `self._fmt_output_data`: Array to store the formatted output data.
        """
        for (didx, data) in enumerate(self.raw_data):
            dm, dn = np.shape(data)
            fm, fn = self._expanded_shapes[didx]
            m1, n1 = (fm - dm) // 2, (fn - dn) // 2
            m2, n2 = fm - dm - m1, fn - dn - n1
            expanded_output = np.pad(data, ((m1, m2), (n1, n2)), 'wrap')
            expanded_output = self._rotate_then_crop_expanded_data(expanded_output, didx)

            expanded_shape = np.shape(expanded_output)
            expanded_input = np.zeros(expanded_shape, dtype=self._data_type)
            topo_idx = expanded_output < 0
            expanded_input[topo_idx] = -expanded_output[topo_idx]
            expanded_output[topo_idx] = 0
            icur_slice = self._slice_indices[didx].start
            fm, fn = self.fmt_shape
            for i in range(expanded_shape[0] // fm):
                for j in range(expanded_shape[1] // fn):
                    mbeg_idx, mend_idx = i * fm, (i + 1) * fm
                    nbeg_idx, nend_idx = j * fn, (j + 1) * fn
                    self._fmt_input_data[icur_slice, 0, :, :] = expanded_input[mbeg_idx:mend_idx, nbeg_idx:nend_idx]
                    self._fmt_output_data[icur_slice, 0, :, :] = expanded_output[mbeg_idx:mend_idx, nbeg_idx:nend_idx]
                    icur_slice += 1

    def _restore_single_raw_output_data_from_slices(self, data_idx, slices, slice_start_idx, fill_value):
        """
        Restore a single raw output data array from its slices.

        Parameters:
        data_idx (int): Index of the data to be restored.
        slices (np.ndarray): Array of slices to be used for restoration.
        slice_start_idx (int): Starting index of the slices.
        fill_value (numeric): Value to fill in the raw output where the original data is negative.

        Returns:
        np.ndarray: The restored raw output data array.
        """
        nm, nn = self._rotated_blocks[data_idx]
        fm, fn = self.fmt_shape
        expanded_shape = (nm * fm, nn * fn)
        expanded_output = np.empty(expanded_shape, dtype=self._data_type)
        cur_slice_idx = slice_start_idx
        for i in range(expanded_shape[0] // fm):
            for j in range(expanded_shape[1] // fn):
                mbeg_idx, mend_idx = i * fm, (i + 1) * fm
                nbeg_idx, nend_idx = j * fn, (j + 1) * fn
                expanded_output[mbeg_idx:mend_idx, nbeg_idx:nend_idx] = slices[cur_slice_idx, 0, :, :]
                cur_slice_idx += 1
        raw_output = self._extract_single_raw_output(data_idx, expanded_output)
        raw_output[self.raw_data[data_idx] < 0] = fill_value

        return raw_output

    def _rotate_then_crop_expanded_data(self, expanded_data, data_idx):
        """
        Rotate the expanded data by the angle specified for the given data index, 
        then crop the rotated data to fit the desired format shape.

        Parameters:
        expanded_data (numpy.ndarray): The data to be rotated and cropped.
        data_idx (int): The index of the data which determines the rotation angle 
                        and the block size for cropping.

        Returns:
        numpy.ndarray: The rotated and cropped data.
        """
        alpha = self._actual_angles[data_idx]
        if alpha == 0:
            return expanded_data
        expanded_data = ndimage.rotate(expanded_data, alpha, axes=(0, 1), reshape=True, order=0, mode='grid-wrap')
        em, en = np.shape(expanded_data)
        fm, fn = self.fmt_shape
        bm, bn = self._rotated_blocks[data_idx]
        sm, sn = (em - fm * bm) // 2, (en - fn * bn) // 2
        return expanded_data[sm:(sm + fm * bm), sn:(sn + fn * bn)]

    def _extract_single_raw_output(self, data_idx, expanded_output):
        """
        Extracts a single raw output from the expanded output by rotating and cropping.

        Parameters:
        data_idx (int): Index of the data to be extracted.
        expanded_output (ndarray): The expanded output array to be processed.

        Returns:
        ndarray: The cropped and rotated output array.

        Notes:
        - The function first determines the shape of the raw data at the given index.
        - It then rotates the expanded output by the negative of the actual angle at the given index.
        - Finally, it crops the rotated output to match the dimensions of the raw data.
        """
        dm, dn = np.shape(self.raw_data[data_idx])
        alpha = -self._actual_angles[data_idx]
        if alpha != 0:
            expanded_output = ndimage.rotate(expanded_output, alpha, axes=(0, 1), reshape=False, order=1, cval=np.nan)
        sm, sn = (expanded_output.shape[0] - dm) // 2, (expanded_output.shape[1] - dn) // 2
        return expanded_output[sm:sm + dm, sn:sn + dn]
