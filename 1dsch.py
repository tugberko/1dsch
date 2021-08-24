#!/usr/bin/python
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np


dx    = 0.02                     # spatial separation
x     = np.arange(0, 10, dx)       # spatial grid points


# Initial Wavefunction

kx    = 0.1 # wave number
m     = 1 # mass
sigma = 0.1 # width of initial gaussian wave-packet
x0    = 3.0 # center of initial gaussian wave-packet

A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)
