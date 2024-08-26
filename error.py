from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pickle
import xspec
import emullver
import plotting
import _globals
from xspec_comparison import comparison_figure
from emullver import params_to_physical_units
from xspec_comparison import compare_spectra
from xspec import *
import torch
import torch.nn as nn
import torch.optim as optim
from xspec_comparison import xspec_spectrum
from emullver import emulated_spectrum
import matplotlib.colors as colors

Xset.chatter = 0
Xset.logChatter = 0

def get_mean_square_error(emulated_spec, xillver_spec):
    """
    Compute mean square error for emulated spectra

    :param emulated_spec: The emulated X-ray spectra
    :param xillver_spec: The XILLVER model X-ray spectra
    :return: The mean square error
    """
    return np.mean(np.square(emulated_spec - xillver_spec))

def create_errors():
    """
    Creates file of array of (gamma, Afe, logxi, Ecut, Incl, error)
    for Matzeu et al emulator
    """
    with open("data.pickle", "rb") as handle:
        data = pickle.load(handle)

    train_x = np.array(data['train_x']).reshape(62400, 5)
    emulator = emullver.load_emulator()
    tuples = []
    count = 0
    for x in train_x:
        phy_x = params_to_physical_units(x)
        gamma, Afe, logxi, Ecut, Incl = map(np.float64, phy_x)
        egrid_e, spec_e, egrid_x, spec_x = map(np.array, compare_spectra(
                                    gamma=gamma,
                                    Afe=Afe,
                                    logxi=logxi,
                                    Ecut=Ecut,
                                    Incl=Incl,
                                    emulator=emulator,
                                    xillver_model=None,
                                    table_path="./xillver-a-Ec4-full.fits",
                                ))
        spec_e = spec_e + 1
        spec_x = spec_x + 1
        spec_e = np.log10(spec_e)
        spec_x = np.log10(spec_x)
        error = get_mean_square_error(spec_e, spec_x)
        tuples.append((gamma, Afe, logxi, Ecut, Incl, error))
        count += 1
        if count % 1000 == 0:
            print(count)
    tuples = np.array(tuples)
    np.save("Emulator_Train_Errors.npy", tuples)

def get_errors_from_file(file):
    """
    Returns array of (gamma, Afe, logxi, Ecut, Incl, error)

    :param str file: Path to the file of errors
    :return: The lists of the tuples
    """
    return np.load(file)

def graph_errors(tuples, vert=True):
    """
    Graph scatterplots of the error in the 5d parameter space

    :param tuples: The tuples of (gamma, Afe, logxi, Ecut, Incl, error)
    """
    
    # map gamma/Afe pairs to logxi, Ecut, Incl, error
    # also keep track of max/min error
    pair_to_quartet = {}
    max_error = 0
    min_error = float('inf')
    for gamma, Afe, logxi, Ecut, Incl, error, i in tuples:
        if vert:
            pair_to_quartet[(gamma, Afe)] = pair_to_quartet.get((gamma, Afe), [])
            pair_to_quartet[(gamma, Afe)].append([logxi, Ecut, Incl, error])
        else:
            pair_to_quartet[(Afe, gamma)] = pair_to_quartet.get((Afe, gamma), [])
            pair_to_quartet[(Afe, gamma)].append([logxi, Ecut, Incl, error])
        if error != float('inf'):
            max_error = max(max_error, error)
        min_error = min(min_error, error)
    
    # sort gamma/Afe pairs
    for key in pair_to_quartet:
        pair_to_quartet[key] = np.array(pair_to_quartet[key])
    key_pairs = list(pair_to_quartet.keys())
    key_pairs.sort()

    i, j = 0, 0
    while j < len(key_pairs):
        # get indices for each row
        while j < len(key_pairs) and key_pairs[i][0] == key_pairs[j][0]:
            j += 1
        n_graphs = j - i
        
        # setup figures
        if vert:
            fig = plt.figure(figsize=(14, 14))
        else:
            fig = plt.figure(figsize=(56, 14))
        
        # plot each row
        for k in range(i, j):
            # setup parameters
            n = k-i+1
            ax = fig.add_subplot(1, n_graphs, n, projection='3d')
            quartet_list = pair_to_quartet[key_pairs[k]]
            logxi_list = quartet_list[:, 0]
            Ecut_list = quartet_list[:, 1]
            Incl_list = quartet_list[:, 2]
            error_list = quartet_list[:, 3]
            
            if vert:
                gamma = key_pairs[k][0]
                Afe = key_pairs[k][1]
            else:
                gamma = key_pairs[k][1]
                Afe = key_pairs[k][0]
            
            # plot parameters + error
            sc = ax.scatter3D(logxi_list, Ecut_list, Incl_list, c=error_list, cmap='viridis', norm=colors.LogNorm(vmin=min_error, vmax=max_error))
            ax.set_xlabel(r"$\log(\xi)$")
            ax.set_ylabel(r"$E_{\rm cut}$")
            ax.set_zlabel(r"$i$")
            ax.zaxis.labelpad = -0.7
            
            # add text
            pars = (
                r"$\Gamma=%.2f$" % gamma
                + "\n"
                + r"$\rm A_{Fe}=%.2f$" % Afe
            )
            ax.text2D(0.05, 0.95, pars, transform=ax.transAxes, fontsize=12)
            
            # add error bar
            if k == j-1:
                if vert:
                    cbar_ax = fig.add_axes([0.95, 0.4, 0.02, 0.2])
                else:
                    cbar_ax = fig.add_axes([0.92, 0.4, 0.01, 0.2])
                fig.colorbar(sc, ax=ax, cax=cbar_ax, label='Error')
                
        plt.show()
        i = j

def get_spectra():
    """
    Returns array of parameters and array of corresponding spectra

    :return: the parameters (gamma, Afe, logxi, Ecut, Incl, error), and the corresponding X-ray spectra
    """
    hdul = fits.open('xillver-a-Ec4-full.fits')
    spectra = []
    params = []
    for x in hdul[3].data:
        param = np.array(x[0])
        spectrum = np.array(x[1])
        spectra.append(spectrum)
        params.append(param)
    params = np.array(params)
    spectra = np.array(spectra)
    return params, spectra

def get_energy_bins():
    """
    Return the 4999 energy bins

    :return: List of the 4999 energies
    """
    hdul = fits.open('xillver-a-Ec4-full.fits')
    bins = []
    for left, right in hdul[2].data:
        bins.append((left+right) / 2)
    return bins
