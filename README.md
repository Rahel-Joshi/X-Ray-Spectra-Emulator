
# Emulating X-Ray Spectroscopy Utilizing Machine Learning

X-ray observations of astronomical objects like black holes and neutron stars allow us to constrain, with parameters, the most energetic processes in the universe. Currently, physics models, like XILLVER, can simulate X-ray emission from compact objects, allowing astrophysicists to create tables of template X-ray spectra resulting from different combinations of input physical parameters, used to fit observed spectra. Since a perfect fit is unlikely, the current standard is to linearly interpolate the closest spectra. However, due to the non-linear nature of X-ray spectra, linear interpolation can result in inaccurate results. Here, machine learning can be a powerful alternative approach due to its ability to capture non-linearity. Previous studies demonstrate the potential use of neural networks for this use case. In this project, we investigate how neural networks perform in emulating complex spectra with rich emission line signals, exploring different network architectures and data preprocessing techniques. We then compare the results of emulation with standard interpolation techniques evaluating the potential inaccuracies with the conventional approach.



## Background

![alt text](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Example.png)

Features of X-Ray spectra correspond to certain characteristics of a black hole or neutron star system. For example, emission lines like the peak shown by the green line caused by reflections of the accretion disk can tell us a lot about the composition of said accretion disk! The spectra's features also correspond to other parameters such as the inclination of the system, or its ionization, etc. Currently, astrophysicts use models like XILLVER to fit observed X-Ray spectra from telescopes to then derive information regarding the physical parameters (composition, ionization, inclination, etc) of the black hole/neutron star system, through just telescope data! However, the tables use linear interpolation for the fitting, which can result in inaccuracy which we seek to solve through a machine learning approach instead.
<img src="https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Matzeu.png" width="48">
![alt text](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Matzeu.png)

![alt text](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Matzeu2.png)

Matzeu et al, X-Ray Accretion Disk-wind Emulator, 2022 demonstrated that a neural network model can provide more accurate results than current techniques of linear interpolation. However, the Matzeu emulator also appears to have some issues with false emission liens when retrained on the XILLVER table. We seek to develop an alternative emulator that does not these issues with fake emission lines. Take a look at [Model_demo.ipynb](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Model_demo.ipynb) to see how our model performs against the Matzeu emulator (our model has much less noise and error, and thus reduced fake emission lines!)





