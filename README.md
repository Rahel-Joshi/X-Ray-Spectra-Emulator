
# Emulating X-Ray Spectroscopy Utilizing Machine Learning

Link for [SURF Presentation Slides](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/SURF.pdf).

Link for [SURF Proposal](https://rahel-joshi.github.io/SURF_2024_Proposal.pdf).

## Abstract

X-ray observations of astronomical objects like black holes and neutron stars allow us to constrain, with parameters, the most energetic processes in the universe. Currently, physics models, like XILLVER, can simulate X-ray emission from compact objects, allowing astrophysicists to create tables of template X-ray spectra resulting from different combinations of input physical parameters, used to fit observed spectra. Since a perfect fit is unlikely, the current standard is to linearly interpolate the closest spectra. However, due to the non-linear nature of X-ray spectra, linear interpolation can result in inaccurate results. Here, machine learning can be a powerful alternative approach due to its ability to capture non-linearity. Previous studies demonstrate the potential use of neural networks for this use case. In this project, we investigate how neural networks perform in emulating complex spectra with rich emission line signals, exploring different network architectures and data preprocessing techniques. We then compare the results of emulation with standard interpolation techniques evaluating the potential inaccuracies with the conventional approach.



## Background

![alt text](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Example.png)
<p align="center"> Figure 1

Features of X-Ray spectra correspond to certain characteristics of a black hole or neutron star system. For example in Figure 1, emission lines like the peak shown by the green line caused by reflections of the accretion disk can tell us a lot about the composition of said accretion disk! The spectra's features also correspond to other parameters such as the inclination of the system, or its ionization, etc. Currently, astrophysicts use models like [XILLVER](https://sites.srl.caltech.edu/~javier/xillver/) to fit observed X-Ray spectra from telescopes to then derive information regarding the physical parameters (composition, ionization, inclination, etc) of the black hole/neutron star system, through just telescope data! However, the tables use linear interpolation for the fitting, which can result in inaccuracy which we seek to solve through a machine learning approach instead.

<div align="center" style="display:flex">
        <div style="display:flex">
                <img src="https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Matzeu.png" width="50%">
                <p>Figure 2</p>
        </div>
         <div style="display:flex"">
                <img src="https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Matzeu2.png" width="70%">
                <p>Figure 3</p>
        </div>
</div>

Matzeu et al, X-Ray Accretion Disk-wind Emulator, 2022 (Figure 2) demonstrated that a neural network model can provide more accurate results than current techniques of linear interpolation. However, the Matzeu emulator also appears to have some issues with false emission lines (the red line dip in Figure 3 at ~8 keV) when retrained on the XILLVER table, which is problematic as it can indicate incorrect composition of the black hole system. We seek to develop an alternative emulator that does not have these issues with fake emission lines. Take a look at [Model_demo.ipynb](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Model_demo.ipynb) to see how our model performs against the Matzeu emulator (our model has much less noise and error!)





