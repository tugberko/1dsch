#!/usr/bin/python
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la



# Simulation Parameters

dt = 1e-2   # Temporal seperation

n= 1001 # Spatial grid points

L = 50     # Spatial size, symmetric with respect to x=0

# Derived Simulation Parameters

dx = L/(n-1) # Spatial seperation

print("dx = " + str(dx) +"\n")
print("1/(dx^2) = " + str(1/(dx**2)) +"\n")

halfdt = dt*0.5 # Name says it all

kappa = 2*(np.pi)/dx # Upper limit for force constant (k).

x = np.linspace(-0.5*L, 0.5*L, n)    # Spatial grid points


k = L / 50






# Utility functions


# This function outputs an identity matrix which is
# compatible with the spatial grid points vector (x).
def IdentityMatrix():
    I = np.zeros((x.size, x.size))
    for i in range(x.size):
        I[i][i] = 1
    return I

# This function defines the potential as a function of
# position.
def Potential(pos):
    return 0.5*(pos**2)

# This function outputs a matrix representing the kinetic
# energy part of the Hamiltonian (Eqn.13).
def TMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size):
        result[i][i]=-2
    for i in range(x.size-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return (-1/(2*(dx**2))) * result

# This function outputs a matrix representing the potential
# energy part of the Hamiltonian (Eqn.13).
def VMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size):
        result[i][i]=Potential(x[i])
    return result

# This function returns a Hamiltonian matrix compatible with
# the wavefunction.
def HMatrix():
    return TMatrix() + VMatrix()


# This function outputs the left hand side multiplier
# in the Eqn. 17
#
# TODO: Pade approximation?
def LHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2)

    return first_term + second_term

# This function outputs the right hand side multiplier
# in the Eqn. 17
#
# TODO: Pade approximation?
def RHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2)

    return first_term - second_term



LHS = LHS()
RHS = RHS()
InverseOfLHS = np.linalg.inv(LHS)

# U is the time evolution matrix. To keep things time efficient,
# I initialize this matrix once and use it as necessary.
U = np.matmul(InverseOfLHS, RHS)

# This function applies time evolution to a given wavefunction.
def Evolve(some_psi):
    return np.matmul(U, some_psi)


# This function evaluates the normalization integral.
def SquareSum(psi):
    sum = 0
    for i in range(x.size):
        sum += dx * np.abs(psi[i])**2
    return sum


# This function returns the excited state wavefunction for the specified order.
# order = 0 corresponds to the groundstate, order = 1 corresponds to the 1st
# excited state vice versa.
def ExcitedState(order):
    H = HMatrix()
    val,vec=la.eig(H)
    z = np.argsort(val)

    psi = vec[:,z[order]]

    return psi


# This is an implementation of Eqn. 14 in
# https://www.overleaf.com/project/614c914c7242ebd307c3b49c
def GroundStateExact():
    psi = np.zeros((x.size),dtype=complex)
    for i in range(x.size):
        psi[i] = (np.pi**(-0.25))*np.exp(-0.5*(x[i]**2))

    return psi


# This function returns the coherent state wavefunction for the specified force
# constant (k).
def CoherentStateNumerical():
    gs = ExcitedState(0)
    psi = np.zeros((x.size),dtype=complex)

    for i in range(x.size):
        psi[i] = np.exp(1j*k*x[i]) * gs[i]

    return psi


# Exact solution of the coherent state
def CoherentStateExact(t):
    return ((1/np.pi)**(0.25)) * np.exp(   -0.5 * (x-k*np.sin(t))**2   ) * np.exp(1j*k*np.cos(t))


# This function calculates the overlap between two wavefunctions.
def Overlap(psi1, psi2):
    overlap = 0
    for i in range(x.size):
        overlap += psi1[i]*np.conj(psi2[i])*dx
    return overlap

# This function normalizes a given wavefunction
def NormalizeWavefunction(some_psi):
    rescale_by = (1/SquareSum(some_psi))**0.5
    return rescale_by*some_psi




# 𝔻𝔼𝕄𝕆 𝕊𝔼ℂ𝕋𝕀𝕆ℕ

def PsiSquares():
    #timesteps = 1
    l_over_k = L/k

    psi_num = NormalizeWavefunction(CoherentStateNumerical())
    psi_exact = NormalizeWavefunction(CoherentStateExact(0))

    ov = Overlap(psi_num, psi_exact)
    plt.plot(x, np.abs(psi_num)**2, "b-", label=r"$|\Psi(x,t)|^2\;(numerical)$")
    plt.plot(x, np.abs(psi_exact)**2, "r--", label=r"$|\Psi(x,t)|^2\;(exact)$")
    plt.legend()
    plt.grid()
    plt.title( '    Overlap = {:.8f}'.format(np.abs(ov)) + '    L/k = {:.2f}'.format(l_over_k))
    plt.show()


def PsiReals():
    #timesteps = 1
    l_over_k = L/k

    psi_num = NormalizeWavefunction(CoherentStateNumerical())
    psi_exact = NormalizeWavefunction(CoherentStateExact(0))

    ov = Overlap(psi_num, psi_exact)
    plt.plot(x, np.real(psi_num)**2, "b-", label=r"$\Re[\Psi(x,t)]\;(numerical)$")
    plt.plot(x, np.real(psi_exact)**2, "r--", label=r"$\Re[\Psi(x,t)]\;(exact)$")
    plt.legend()
    plt.grid()
    plt.title( '    Overlap = {:.8f}'.format(np.abs(ov)) + '    L/k = {:.2f}'.format(l_over_k))
    plt.show()


def PsiImags():
    #timesteps = 1
    l_over_k = L/k

    psi_num = NormalizeWavefunction(CoherentStateNumerical())
    psi_exact = NormalizeWavefunction(CoherentStateExact(0))

    ov = Overlap(psi_num, psi_exact)
    plt.plot(x, np.imag(psi_num)**2, "b-", label=r"$\Im[\Psi(x,t)]\;(numerical)$")
    plt.plot(x, np.imag(psi_exact)**2, "r--", label=r"$\Im[\Psi(x,t)]\;(exact)$")
    plt.legend()
    plt.grid()
    plt.title( '    Overlap = {:.8f}'.format(np.abs(ov)) + '    L/k = {:.2f}'.format(l_over_k))
    plt.show()

#PsiSquares()
PsiReals()
#PsiImags()
