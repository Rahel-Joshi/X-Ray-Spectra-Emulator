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
from emullver import *
import tensorflow as tf
import matplotlib.colors as colors
from models import *

SPECTRA_FACTOR = 2.720513452838532

Xset.chatter = 0
Xset.logChatter = 0

train_X, train_Y, valid_X, valid_Y, test_X, test_Y = get_data()

train_Y = train_Y / SPECTRA_FACTOR
valid_Y = valid_Y / SPECTRA_FACTOR

model = EncoderDecoderModel()
train_model(model, 800, 64, 0.001, 'encoderdecoder', train_Y, train_Y, valid_Y, valid_Y)