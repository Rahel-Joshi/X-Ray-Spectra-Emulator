"""emullver.py

Collection of xillver emulator tools
"""

import numpy as np
import tensorflow as tf

import _globals


def emulated_spectrum(
    gamma=2.0, Afe=1.0, logxi=3.1, Ecut=300.0, Incl=30.0, emulator=None
):
    """Returns an emulated XILLVER spectrum for energy range
    [1e-4, 1000.1] keV for a given choice of input parameters
    in units of photons / cm^2 / s / keV

    Args:
        -- gamma (float, optional): power law index of the incident
            spectrum in the range [0, 3.4]. Defaults to 2.0.
        -- Afe (float, optional): iron abundance in units of solar
            abundance in the range [0.5,10]. Defaults to 1.0.
        -- logxi (float, optional): ionisation of the accretion disc
            in range [0, 4.7] (neutral to heavily ionised).
            Defaults to 3.1.
        -- Ecut (float, optional): observed high-energy cutoff of the
            primary spectrum in units of keV in the range [20, 1000].
            Defaults to 300.
        -- Incl (float, optional): inclination towards the system wrt
            the normal to the accretion disc in the range [18, 87].
            Defaults to 30.
        -- emulator (Keras model instance, optional): NN emulator
            for XILLVER. Defaults to None.

    Returns:
        -- np.ndarray, shape=(4999,): array of energy bin midpoint
            values in units of keV
        -- np.ndarray, shape=(4999,): emulated XILLVER spectrum
    """
    if emulator is None:
        emulator = load_emulator()

    param_vals = np.array([gamma, Afe, logxi, Ecut, Incl])

    # parsing input as a tensorflow.tensor object of shape
    # (1,5), as required by the emulator
    input = tf.reshape(
        tf.convert_to_tensor(params_to_code_units(param_vals)),
        [1, param_vals.size],
    )

    spectrum = emulator.predict(input, verbose=0)[0]
    out = np.power(10, spectrum / _globals.PREPROCESSING_FACTOR)

    return _globals.ENERGY_GRID, out


def load_emulator(path=_globals.EMULATOR_PATH):
    """Helper function for loading the saved emulator with Tensorflow

    Args:
        -- path (str, optional): path to the saved emulator object.
            Defaults to EMULATOR_PATH.

    Returns:
        Keras model instance: feed-forward network model with pre-trained
            weights and biases, ready for inference / prediction
    """

    return tf.keras.models.load_model(path)


def params_to_physical_units(params, ranges=_globals.FEATURE_RANGES):
    """Takes rescaled emulator input features
    and returns physical parameter values for
    XILLVER

    Args:
        -- params (np.array, shape=(5,)): emulator
            input feature values. Parameters in order:
            Gamma, Afe, logxi, Ecut, Incl.
        -- ranges (np.array, shape=(5,2)): allowable
            ranges of the XILLVER input parameters.
            Defaults to FEATURE_RANGES.

    Returns:
        np.ndarray, shape=(5,): corresponding XILLVER
            input parameter values

    """
    out = params * (np.diff(ranges, axis=1).flatten()) + ranges[:, 0]
    return out


def params_to_code_units(params, ranges=_globals.FEATURE_RANGES):
    """Takes physical input parameter values for
    XILLVER and returns rescaled emulator input
    feature values

    Args:
        -- params (np.array, shape=(5,)): physical
            XILLVER input parameter values
        -- ranges (np.array, shape=(5,2)): allowable
            ranges of the XILLVER input parameters.
            Defaults to FEATURE_RANGES.

    Returns:
        np.ndarray, shape=(5,): corresponding emulator
            input feature values
    """
    out = np.divide(params - ranges[:, 0], np.diff(ranges, axis=1).flatten())
    return out
