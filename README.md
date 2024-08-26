
# Emulating X-Ray Spectroscopy Utilizing Machine Learning

X-ray observations of astronomical objects like black holes and neutron stars allow us to constrain, with parameters, the most energetic processes in the universe. Currently, physics models, like XILLVER, can simulate X-ray emission from compact objects, allowing astrophysicists to create tables of template X-ray spectra resulting from different combinations of input physical parameters, used to fit observed spectra. Since a perfect fit is unlikely, the current standard is to linearly interpolate the closest spectra. However, due to the non-linear nature of X-ray spectra, linear interpolation can result in inaccurate results. Here, machine learning can be a powerful alternative approach due to its ability to capture non-linearity. Previous studies demonstrate the potential use of neural networks for this use case. In this project, we investigate how neural networks perform in emulating complex spectra with rich emission line signals, exploring different network architectures and data preprocessing techniques. We then compare the results of emulation with standard interpolation techniques evaluating the potential inaccuracies with the conventional approach.



## Background

![alt text]([http://url/to/img.png](https://github.com/Rahel-Joshi/X-Ray-Spectra-Emulator/blob/main/Example.png))

Features of X-Ray spectra correspond to certain characteristics of a black hole or neutron star system. For example, emission lines like the peak shown by the green line caused by reflections of the accretion disk can tell us a lot about the composition of said accretion disk! The spectra's features also correspond to other parameters such as the inclination of the system, or its ionization, etc. Currently, astrophysicts use models like XILLVER to fit observed X-Ray spectra from telescopes to then derive information regarding the physical parameters (composition, ionization, inclination, etc) of the black hole/neutron star system, through just telescope data! However, the tables use linear interpolation for the fitting, which can result in inaccuracy which we seek to solve through a machine learning approach instead.


