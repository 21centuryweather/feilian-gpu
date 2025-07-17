import csv
import os
import re
import sys
from datetime import datetime
from os import walk

import numpy as np
import torch.nn as nn

from feilian import DataFormatter, FeilianNet, train_network_model_with_adam, predict_with_model

PATTERN = re.compile(r"/([^/]+)_deg(\d+)(.?)\.npy")


def load_files_from_path(datapath):
    """
    Load .npy files from a specified path, handling different directory structures.

    This function generates a list of file paths for .npy files based on the provided
    datapath. It handles specific directory structures by expanding the paths accordingly.

    Args:
        datapath (str): The base path to search for .npy files. It can be a specific path
                        or a pattern that includes "data/all" or "/all".

    Returns:
        list: A sorted list of file paths for .npy files found in the specified directories.
    """
    paths = []
    if datapath.startswith("data/all"):
        paths.extend([f"data/{x}/{y}/" for x in ["uniform", "variable"] for y in ["realistic", "idealised"]])
    elif "/all" in datapath:
        paths.extend([datapath.replace("/all", "/idealised"), datapath.replace("/all", "/realistic")])
    else:
        paths.append(datapath)
    datafiles = []
    for path in paths:
        for (_, _, fnames) in walk(path):
            datafiles.extend([os.path.join(path, fname) for fname in fnames if fname.endswith(".npy")])
            break
    datafiles.sort()
    return datafiles


def parse_wind_angle(filename):
    """
    Parses the wind angle from the given filename using a predefined pattern.

    Args:
        filename (str): The name of the file to parse.

    Returns:
        int: The parsed wind angle if the pattern matches, otherwise 0.
    """
    rematch = PATTERN.search(filename)
    if rematch:
        return int(rematch.group(2))
    return 0


def parse_topo_name(filename):
    """
    Parses the topology name from the given filename using a predefined pattern.
    
    Args:
        filename (str): The name of the file to parse.
        
    Returns:
        str: The parsed topology name if the pattern matches, otherwise a default
             name based on the current timestamp in the format 'topoHH-MM-SS.ffffff'.
    """
    rematch = PATTERN.search(filename)
    if rematch:
        return rematch.group(1)
    return f"topo{datetime.now().strftime('%H-%M-%S.%f')}"


def write_dicts_to_csv(data, filenames, case_angles, csvname):
    """
    Writes data from dictionaries to a CSV file.

    Parameters:
    data (dict): A dictionary where keys are column headers and values are lists of column data.
    filenames (list of str): A list of filenames to be parsed for topological names.
    case_angles (list of int or float): A list of angles corresponding to each case.
    csvname (str): The name of the CSV file to write the data to.

    Returns:
    None
    """
    topo = [parse_topo_name(fn) for fn in filenames] + ["total"]
    caseangles = case_angles + [0]
    with open(csvname, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["topo", "angle"] + list(data.keys()))
        csv_writer.writerows(zip(topo, caseangles, *data.values()))


# qsub -I -lwalltime=10:00:00,ncpus=24,ngpus=2,mem=64GB,wd -qgpuvolta -Ppu02
if __name__ == "__main__":
    case_type = "variable/idealised/"  # "all/"
    data_path = f"data/{case_type}"
    files = load_files_from_path(data_path)
    images = [np.load(f) for f in files]

    angles = [int(PATTERN.search(fname).group(2)) for fname in files]

    data_fmt = DataFormatter(images, wind_angles=angles, formatted_shape=1280)
    seed = int(sys.argv[1])
    x_train, y_train, train_idx, x_test, y_test, test_idx = data_fmt.split_train_test_data(0.8, seed)
    print(f"data split seed:\t{seed}")
    print(f"train indices:\t{train_idx}")
    print(f"train data shape:\t{np.shape(x_train)}")
    print(f"test indices:\t{test_idx}")
    print(f"test data shape:\t{np.shape(x_test)}")

    activation, batch_size = nn.ReLU(inplace=True), 4
    model = FeilianNet(chan_multi=20, max_level=6, activation=activation)
    print(f"number of model parameters: {model.count_trainable_parameters()}")
    print(f"activation is: {str(activation).split('(')[0]}")
    print(f"batch size is: {batch_size}")
    model = train_network_model_with_adam(model, x_train, y_train, batch_size=batch_size,
                                          model_dir="/scratch/pu02/wl0925/feilian/models")

    y_train_pred = predict_with_model(model, x_train, batch_size)
    y_test_pred = predict_with_model(model, x_test, batch_size)

    def save_training_results(path, fmt_pred, indices, csv_name, saveimages=False):
        """
        Save training results including metrics and optionally images.

        Parameters:
        path (str): The directory path where results will be saved.
        fmt_pred (array-like): Formatted predictions from the model.
        indices (list): List of indices corresponding to the data samples.
        csv_name (str): The name of the CSV file to save the metrics.
        saveimages (bool, optional): If True, save the prediction and truth images. Defaults to False.

        Returns:
        None
        """
        if not os.path.exists(path):
            os.makedirs(path)
        metrics_csv = f"{path}/{csv_name}"
        metrics = data_fmt.compute_all_metrics(fmt_pred, indices)
        write_dicts_to_csv(metrics, [files[i] for i in indices], [angles[i] for i in indices], metrics_csv)

        if not saveimages:
            return
        raw_pred = data_fmt.restore_raw_output_data(fmt_pred, indices, np.nan)
        for (pidx, tidx) in enumerate(indices):
            pred, truth = np.rot90(raw_pred[pidx], k=-(angles[tidx] // 90)), images[tidx]
            truth[truth < 0] = np.nan
            casename = files[tidx].split("/")[-1].split(".npy")[0]
            np.save(f"{path}/{casename}_truth.npy", truth)
            np.save(f"{path}/{casename}_prediction.npy", pred)

    curr_time = datetime.now().strftime('%Y%m%dT%H:%M:%S.%f')
    train_images_dir = f".output/images/{case_type}/train"
    train_metrics_csv = f"-metrics_in_training_set_seed{seed}_act{str(activation).split('(')[0]}_time{curr_time}.csv"
    save_training_results(train_images_dir, y_train_pred, train_idx, train_metrics_csv)

    test_images_dir = f".output/images/{case_type}/test"
    test_metrics_csv = f"-metrics_in_test_set_seed{seed}_act{str(activation).split('(')[0]}_time{curr_time}.csv"
    save_training_results(test_images_dir, y_test_pred, test_idx, test_metrics_csv)
