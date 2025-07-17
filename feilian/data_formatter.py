import math

import numpy as np
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DEFAULT_FORMAT_SHAPE = 1280


def _find_formatted_shape(fmt_shape):
    if isinstance(fmt_shape, int):
        return fmt_shape, fmt_shape
    if isinstance(fmt_shape, (tuple, list)):
        if len(fmt_shape) == 1:
            return fmt_shape[0], fmt_shape[0]
        return fmt_shape[0], fmt_shape[1]
    return DEFAULT_FORMAT_SHAPE, DEFAULT_FORMAT_SHAPE


def _determine_shape_before_rotation(alpha, fmt_shape):
    ralpha = np.deg2rad(alpha)
    r = np.array([[np.cos(ralpha), np.sin(ralpha)], [np.sin(ralpha), np.cos(ralpha)]])
    tshape = np.linalg.solve(r, np.array(fmt_shape))
    if np.any(tshape <= 0):
        max_shape = np.max(fmt_shape)
        tshape = np.linalg.solve(r, np.array([max_shape, max_shape]))
    return tshape


def _is_point_below_line(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    return k * point[0] + b >= point[1]


def _compute_expanded_shape_and_rotated_blocks(data, alpha, fmt_shape):
    """
    Compute the shapes of the expanded and formatted images for data.
    The following is the meaning of the prefix:
    - d: data
    - f: formatted
    - t: target
    - n: number
    - s: start
    - a: actual
    - ef: expanded formatted

    :param data: The input image.
    :param alpha: The angle the image needs to rotate in counter-clockwise direction
    :param fmt_shape: The desired shape.
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
    all_metrics = {"mae": [], "rmse": [], "r2_score": [], "rel_l2_error": [],
                   "true_mean": [], "pred_mean": [], "abs_mean_diff": [], "rel_mean_diff": [],
                   "true_std": [], "pred_std": [],  "abs_std_diff": [], "rel_std_diff": []}
    return all_metrics


def _add_more_metrics(all_metrics, y_true, y_pred):
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
        dm, dn = np.shape(self.raw_data[data_idx])
        alpha = -self._actual_angles[data_idx]
        if alpha != 0:
            expanded_output = ndimage.rotate(expanded_output, alpha, axes=(0, 1), reshape=False, order=1, cval=np.nan)
        sm, sn = (expanded_output.shape[0] - dm) // 2, (expanded_output.shape[1] - dn) // 2
        return expanded_output[sm:sm + dm, sn:sn + dn]
