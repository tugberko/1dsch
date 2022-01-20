#!/usr/bin/python
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
from scipy.fftpack import fft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from numpy import linspace, zeros, array, pi, sin, cos, exp, arange, matmul, abs, conj, real, convolve, ones

plt.rcParams['figure.figsize'] = 16, 9

filename = 'splitstep.dat'

# Simulation Parameters

dt = 0.02   # Temporal seperation

halfdt = dt*0.5

N = 2**8

L = 3 * pi

dx = L/N

x=arange(-L/2,L/2,dx)


# This function generates the momentum space grid.
def GenerateMomentumSpace():
    dk = (2*pi/L)
    k = zeros(N)
    if ((N%2)==0):
        #-even number
        for i in range(1,N//2):
            k[i]=i
            k[N-i]=-i
    else:
        #-odd number
        for i in range(1,(N-1)//2):
            k[i]=i
            k[N-i]=-i

    return dk * k

k = GenerateMomentumSpace()

ksq = k**2



# This function produces a discrete 1D Laplacian matrix compatible with wavefunction.
def laplacian1D():
    result = np.zeros((N, N))
    for i in range(N):
        result[i][i]=-2
    for i in range(N-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return 1/(dx**2) * result

# This function outputs a matrix representing the kinetic
# energy part of the Hamiltonian.
def TMatrix():
    return -0.5 * laplacian1D()


# This function generates an array of potential energies corresponding to the
# position space.
def Potential():
    pot = np.zeros_like(x)
    for i in range(N):
        pot[i] = 0.5*(x[i]**2)
    return pot


# This function outputs a matrix representing the potential
# energy part of the Hamiltonian.
def VMatrix():
    return np.diag(Potential())

# This function creates a discrete 1D kinetic energy matrix.
def TMatrix():
    return -0.5 * laplacian1D()

# This function creates a Hamiltonian matrix.
def HMatrix():
    return TMatrix() + VMatrix()

# This function creates the time evolution matrix (U) for Crank-Nicholson method.
def InitializeCrankNicholsonU():
    lhs = la.expm(1j * HMatrix() * halfdt)
    rhs = la.expm(-1j * HMatrix() * halfdt)
    return np.matmul(la.inv(lhs) , rhs)


UCN = InitializeCrankNicholsonU()

# This function performs Crank-Nicholson time evolution
def EvolveCrankNicholson(some_psi):
    return np.matmul(UCN, some_psi)



V = VMatrix()


# This function performs split step fourier time evolution.
def EvolveSplitStep(some_psi):
    # Refer to: https://www.overleaf.com/7461894969pxkgqkzvmdws

    # First, neglect kinetic for half step, U1
    psi = matmul( la.expm( -1j * halfdt * V) , some_psi  )

    # Second, neglect potential for whole step, U2-hat in momentum space
    psi_hat = fft(psi)
    psi_hat = np.exp( -1j * ksq * dt / 2) * psi_hat
    psi = ifft(psi_hat)

    # Third, neglect kinetic for half more step, U3
    psi = matmul( la.expm( -1j * halfdt * V) , psi  )

    return psi




# Exact solution of the coherent state
def CoherentStateExact(t):
    k=1
    return ((1/np.pi)**(0.25)) * np.exp(   -0.5 * (x-k*np.sin(t))**2   ) * np.exp(1j*k*x*np.cos(t))

# This function calculates the overlap between two wavefunctions.
def Overlap(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        overlap += psi1[i] * np.conj(psi2[i]) * dx
    return np.abs(overlap)





# DEMO SECTION

psi_num_splitstep = CoherentStateExact(0)
psi_num_cn = CoherentStateExact(0)

terminateAt = 20 * np.pi
timesteps = math.ceil(terminateAt / dt)

for i in range(timesteps):
    currentTime = i*dt

    psi_exact = CoherentStateExact(currentTime)

    current_time_as_string = "{:.5f}".format(currentTime)
    print('Current time: ' + current_time_as_string)


    plt.suptitle('Grid Size: ' +str(N)+'\nTime: ' + current_time_as_string)
    plt.subplot(1, 3, 1)
    plt.plot(x, np.abs(psi_num_splitstep)**2)
    error_ss = np.abs(1 - Overlap(psi_exact, psi_num_splitstep))
    error_ss_string = "{:.3e}".format(error_ss)
    plt.title('Split-step\nError: '+ error_ss_string)
    plt.ylim(-0.02 , 0.6)
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(x, np.abs(psi_num_cn)**2)
    error_cn = np.abs(1 - Overlap(psi_exact, psi_num_cn))
    error_cn_string = "{:.3e}".format(error_cn)
    plt.title('Crank-Nicholson\nError: '+ error_cn_string)
    plt.ylim(-0.02 , 0.6)
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(x, np.abs(psi_exact)**2)
    plt.title('Exact solution')
    plt.ylim(-0.02 , 0.6)
    plt.grid()

    plt.tight_layout()
    plt.savefig('visual'+str(i)+'.png', dpi=120)
    plt.clf()



    psi_num_splitstep = EvolveSplitStep(psi_num_splitstep)
    psi_num_cn = EvolveCrankNicholson(psi_num_cn)
