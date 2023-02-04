import os
from typing import List

import h5py
import numpy as np
import requests
import urllib3
from tqdm.auto import tqdm

from .d4rl_infos import DATASET_URLS

urllib3.disable_warnings()

global D4RL_DATASET_PATH


def urlretrieve(path, url):
    with open(path, "wb") as f:
        chunk_size = 10 * 1024
        r = requests.get(url, stream=True, verify=False)
        total = int(r.headers.get("content-length"))
        r.raise_for_status()

        for chunk in tqdm(
            r.iter_content(chunk_size),
            total=round(total / chunk_size),
            desc="Downloading",
        ):
            f.write(chunk)


def set_dataset_path(path):
    global D4RL_DATASET_PATH
    D4RL_DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(D4RL_DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urlretrieve(dataset_filepath, dataset_url)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def get_d4rl_dataset(env_name: str, qpos=False):
    dataset_url = DATASET_URLS[env_name]
    h5path = filepath_from_url(dataset_url)
    if not os.path.isfile(h5path):
        h5path = download_dataset_from_url(dataset_url)

    data_dict = {}
    with h5py.File(h5path, "r") as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ["observations", "actions", "rewards", "terminals"]:
        assert key in data_dict, "Dataset is missing key %s" % key

    n_samples = data_dict["observations"].shape[0]

    if data_dict["rewards"].shape == (n_samples, 1):
        data_dict["rewards"] = data_dict["rewards"][:, 0]
    assert data_dict["rewards"].shape == (n_samples,), "Reward has wrong shape: %s" % (
        str(data_dict["rewards"].shape)
    )
    if data_dict["terminals"].shape == (n_samples, 1):
        data_dict["terminals"] = data_dict["terminals"][:, 0]
    assert data_dict["terminals"].shape == (
        n_samples,
    ), "Terminals has wrong shape: %s" % (str(data_dict["terminals"].shape))
    return data_dict


def get_observations_with_qpos(d4rl_dataset, qpos: List[int]):
    """Insert the required qpos in the returned observations.

    :param d4rl_dataset: a d4rl (hdf5) dataset
    :param qpos: list of qpos to insert at same qpos
    :return: new observations (all observations array) with qpos inserted
    """
    new_observations = np.array(d4rl_dataset["observations"])
    for pos in qpos:
        new_observations = np.insert(
            new_observations, pos, d4rl_dataset["infos/qpos"][:, pos], axis=1
        )
    return new_observations


if __name__ == "__main__":
    set_dataset_path("/d4rl-data")
    dataset = get_d4rl_dataset("halfcheetah-medium-v0")