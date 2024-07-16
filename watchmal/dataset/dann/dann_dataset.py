"""
Class implementing a PMT dataset for DANNs in h5 format
Modified from mPMT dataset for use with single PMTs
"""

# torch imports
from torch import from_numpy, Tensor, roll, flip
import torch
import torchvision

# generic imports
import numpy as np

import random

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset, normalize
import watchmal.dataset.data_utils as du

from watchmal.dataset.cnn.cnn_dataset import CNNDataset

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, MaxAbsScaler

import h5py
import joblib

class DANNDataset(CNNDataset):
    def __init__(self, h5file, pmt_positions_file, domain_labels, use_times=True, use_charges=True, use_positions=False, transforms=None, one_indexed=True, channel_scaling=None, geometry_file=None):
        super().__init__(h5file, pmt_positions_file, use_times=use_times, use_charges=use_charges, use_positions=use_positions, transforms=transforms, one_indexed=one_indexed, channel_scaling=channel_scaling, geometry_file=geometry_file)
        self.domain_labels = domain_labels
    
    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        # Add domain label to the data dictionary
        data_dict['domain'] = self.domain_labels[item]

        return data_dict