import pickle

import torch
from einops import rearrange
from torch.utils.data import Dataset

from soft_robot.dataset.dataloader import utils


class SmartwatchDataset(Dataset):
    def __init__(self, dataset_path: str, num_ensemble: float, dim_x, dim_z):
        """
        Constructor
        Args:
            dataset_path: a path to a pickle file. Expects a dictionary with "state_gt", "pre_state", and "obs"
        """

        # load data
        dat = pickle.load(open(dataset_path, "rb"))
        self.__state = torch.tensor(dat["state_gt"], dtype=torch.float32)
        self.__pre_state = torch.tensor(dat["pre_state"], dtype=torch.float32)
        self.__obs = torch.tensor(dat["obs"], dtype=torch.float32)

        self.__xx_m = dat["xx_m"]
        self.__xx_s = dat["xx_s"]
        self.__yy_m = dat["yy_m"]
        self.__yy_s = dat["yy_s"]

        # make use of the soft_robot utils class
        self.__utils = utils(num_ensemble, dim_x, dim_z)

    def get_norm_stats(self):
        return self.__xx_m, self.__xx_s, self.__yy_m, self.__yy_s

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

        state_gt, state_pre, obs = self.__state[idx], self.__pre_state[idx], self.__obs[idx]

        state_gt = rearrange(state_gt, "(k dim) -> k dim", k=1)
        state_pre = rearrange(state_pre, "(k dim) -> k dim", k=1)
        obs = rearrange(obs, "(k dim) -> k dim", k=1)

        state_ensemble = self.__utils.format_state(state_pre, noise=0.02)
        return state_gt, state_ensemble, obs
