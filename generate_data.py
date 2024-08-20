from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import emullver
import plotting
import _globals
from xspec import *
import torch
import torch.nn as nn
import torch.optim as optim
from xspec_comparison import *
from emullver import *
from error import *
import random
from emullver import params_to_code_units
import tensorflow as tf


def create_split_data():
    '''
    save the training, validation, and testing data
    '''
    params, spectra = get_spectra()

    # convert all params to code units, and log the spectra
    # don't include spectra with 0 counts
    phy_params = []
    log_spectra = []
    for param, spectrum in zip(params, spectra):
        if spectrum.min() > 0:
            phy_params.append(params_to_code_units(param))
            log_spectra.append(np.log10(spectrum))
    phy_params = np.array(phy_params)
    log_spectra = np.array(log_spectra)

    # zip together params and spectra, shuffle
    pairs = list(zip(phy_params, log_spectra))
    random.shuffle(pairs)

    # partition the data
    n = len(pairs)
    train_num = int(.8 * n)
    test_num = int(.1 * n)
    valid_num = int(.1 * n)

    # zip together spectra and params separately
    train_x, train_y = zip(*pairs[:train_num])
    test_x, test_y = zip(*pairs[train_num:train_num+test_num])
    valid_x, valid_y = zip(*pairs[train_num+test_num:train_num+test_num+valid_num])

    # save files
    np.save('train_x.npy', train_x)
    np.save('train_y.npy', train_y)
    np.save('test_x.npy', test_x)
    np.save('test_y.npy', test_y)
    np.save('valid_x.npy', valid_x)
    np.save('valid_y.npy', valid_y)