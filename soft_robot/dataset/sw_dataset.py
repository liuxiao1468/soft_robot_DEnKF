import pickle

import torch
from torch.utils.data import Dataset


class SmartwatchDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Constructor
        Args:
            dataset_path: a path to a pickle file. Expects a dictionary with "state_gt", "pre_state", and "obs"
        """
        dat = pickle.load(open(dataset_path, "rb"))
        self.__state = torch.tensor(dat["state_gt"], dtype=torch.float32)
        self.__pre_state = torch.tensor(dat["pre_state"], dtype=torch.float32)
        self.__obs = torch.tensor(dat["obs"], dtype=torch.float32)

    def __len__(self):
        """
        Returns: Length of the Dataset
        """
        return len(self.__obs)

    def __getitem__(self, idx):
        """
        Fetch an item from the Dataset
        Args:
            idx: Index of the requested item
        Returns: state [idx], state_pre [idx], obs[idx]
        """
        return self.__state[idx], self.__pre_state[idx], self.__obs[idx]
