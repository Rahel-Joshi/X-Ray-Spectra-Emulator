import matplotlib.pyplot as plt
import numpy as np
import xspec
from matplotlib.ticker import MultipleLocator

import _globals
import emullver


def compare_spectra(
    gamma,
    Afe,
    logxi,
    Ecut,
    Incl,
    emulator=None,
    xillver_model=None,
    table_path=None,
):
    """Returns energy grid midpoints and spectra
    for both the emulator and an xspec additive table
    model for a given combination of input xillver parmeters.

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
        -- xillver_model (xspec.Model instance, optional): pyxspec
            additve table Model instance with a xillver table.
            Defaults to None.
        -- table_path (str, optional): path to the xillver model
            fits table. Defaults to None.

    Returns:
        -- np.ndarray, shape=(N,): array of energy bin midpoint
            values in units of keV, as extracted from XSPEC
        -- np.ndarray, shape=(N,): XILLVER spectrum interpolated by XSPEC
        -- np.ndarray, shape=(4999,): array of energy bin midpoint
            values in units of keV for the emulator
        -- np.ndarray, shape=(4999,): emulated XILLVER spectrum

    """
    # spectra
    egrid_e, spec_e = emullver.emulated_spectrum(
        gamma=gamma,
        Afe=Afe,
        logxi=logxi,
        Ecut=Ecut,
        Incl=Incl,
        emulator=emulator,
    )
    egrid_x, spec_x = xspec_spectrum(
        gamma=gamma,
        Afe=Afe,
        logxi=logxi,
        Ecut=Ecut,
        Incl=Incl,
        xillver_model=xillver_model,
        table_path=table_path,
    )

    return egrid_e, spec_e, egrid_x, spec_x

def comparison_figure(
    gamma=2.0,
    Afe=1.0,
    logxi=3.1,
    Ecut=300.0,
    Incl=30.0,
    emulator=None,
    xillver_model=None,
    table_path=None,
    gridpoint="NO",
    erange=(2e-4, 1e3),
    zoomrange=(3.0, 10.0),
    savename=None,
):
    """Plots a figure comparing an emulated and xspec
    spectrum for the same set of input xillver parameters.
    When provided with a savename, saves at the given string
    and closes the figure.

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
        -- xillver_model (xspec.Model instance, optional): pyxspec
            additve table Model instance with a xillver table.
            Defaults to None.
        -- table_path (str, optional): path to the xillver model
            fits table. Defaults to None.
        -- gridpoint (str, optional): whether a given set of input
            parameters falls on a xillver table gridpoint. Defaults to "NO".
        -- erange (tuple, optional): energy range for figure plotting.
            Defaults to (2e-4, 1e3).
        -- zoomrange (tuple, optional): energy range for a zoom-in panel
            on the right. Defaults to (3.0, 10.0).
        -- savename (str, optional): when provided, the figure is saved
            at savename. Defaults to None.
    """
    # spectra
    egrid_e, spec_e, egrid_x, spec_x = compare_spectra(
        gamma=gamma,
        Afe=Afe,
        logxi=logxi,
        Ecut=Ecut,
        Incl=Incl,
        emulator=emulator,
        xillver_model=xillver_model,
        table_path=table_path,
    )

    # figure setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_xlim(erange)
    suball = np.where((egrid_x > erange[0]) & (egrid_x < erange[1]))
    ax1.set_ylim(
        0.98 * np.array(spec_e)[suball].min(),
        2 * np.array(spec_e)[suball].max(),
    )

    # data
    for a in (ax1, ax2):
        a.loglog(egrid_x, spec_x, c="k", lw=1, label="xspec", alpha=0.8)
        a.loglog(egrid_e, spec_e, c="r", lw=0.75, label="emulator", alpha=0.5)
        a.set_xlabel("Energy [keV]")
        a.set_ylabel(r"$\rm photons / cm^2 / s / keV$")

    ax1.legend()

    # zoomrange subset
    sub = np.where((egrid_x > zoomrange[0]) & (egrid_x < zoomrange[1]))
    ymax = max(spec_e[sub].max(), np.array(spec_x)[sub].max())
    ymin = min(spec_e[sub].min(), np.array(spec_x)[sub].min())
    ax2.set_ylim(0.9 * ymin, 2 * ymax)
    ax2.set_xlim(zoomrange)
    ax2.set_xscale("linear")

    # parameter values
    pars = (
        r"$\Gamma=%.2f$" % gamma
        + "\n"
        + r"$\rm A_{Fe}=%.2f$" % Afe
        + "\n"
        + r"$\log(\xi)=%.2f$" % logxi
        + "\n"
        + r"$E_{\rm cut}=%.2f$" % Ecut
        + "\n"
        + r"$i=%.2f$" % Incl
    )
    ax1.text(0.1, 0.1, pars, transform=ax1.transAxes, fontsize=10)

    # diagnostics
    diff = spec_e - spec_x
    L2 = np.sqrt(np.power(diff, 2).sum())
    L2_sub = np.sqrt(np.power(diff[sub], 2).sum())
    Linf = np.amax(np.abs(diff))
    Linf_sub = np.amax(np.abs(diff[sub]))

    def as_si(x, ndp):
        s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
        m, e = s.split("e")
        return r"{m:s}\times 10^{{{e:d}}}".format(m=m, e=int(e))

    text = (
        f"gridpoint? {gridpoint}\n"
        + r"$L_2={0:s}$".format(as_si(L2, 2))
        + "\n"
        + r"$L_\infty={0:s}$".format(as_si(Linf, 2))
    )
    ax1.text(0.4, 0.1, text, transform=ax1.transAxes, fontsize=10)

    text_sub = (
        r"$L_2={0:s}$".format(as_si(L2_sub, 2))
        + "\n"
        + r"$L_\infty={0:s}$".format(as_si(Linf_sub, 2))
    )
    ax2.text(0.1, 0.1, text_sub, transform=ax2.transAxes)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))

    # saving figure
    if savename is not None:
        fig.savefig(savename)
        plt.close(fig)


def xspec_spectrum(
    gamma=2.0,
    Afe=1.0,
    logxi=3.1,
    Ecut=300.0,
    Incl=30.0,
    xillver_model=None,
    table_path=None,
):
    """Returns a XILLVER spectrum for a given set of input parameters
    as generated by XSPEC based on xillver-a-Ec4-full.fits table,
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
        -- xillver_model (xspec.Model instance, optional): pyxspec
            additve table Model instance with a xillver table.
            Defaults to None.
        -- table_path (str, optional): path to the xillver model
            fits table. Defaults to None.

    Returns:
        -- np.ndarray, shape=(4999,): array of energy bin midpoint values
        -- np.ndarray, shape=(4999,): XILLVER spectrum interpolated by XSPEC
    """
    if xillver_model is None:
        xspec.AllModels.setEnergies(_globals.ENERGY_BIN_FILE)

        if table_path is None:
            print("XILLVER model table path missing. Please fix!")
        else:
            xillver_model = xspec.Model("atable{%s}" % table_path)

    xillver_model.setPars({1: gamma, 2: Afe, 3: logxi, 4: Ecut, 5: Incl})
    energy_grid = (
        xillver_model.energies(0)[:-1]
        + np.diff(xillver_model.energies(0)) / 2.0
    )

    return energy_grid, xillver_model.values(0)
