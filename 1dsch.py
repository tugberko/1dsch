#!/usr/bin/python
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np



# Simulation Parameters

dt = 1e-3   # Temporal seperation

dx = 2e-2   # Spatial seperation

L = 50      # Spatial size, symmetric with respect to x=0

# Derived Simulation Parameters

halfdt = dt*0.5 # Name says it all

kappa = 2*(np.pi)/dx # Upper limit for force constant (k).

x = np.arange(-0.5*L, 0.5*L, dx)    # Spatial grid points






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
    return (-1/dx**2) * result

# This function outputs a matrix representing the potential
# energy part of the Hamiltonian (Eqn.13).
def VMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size):
        result[i][i]=Potential(x[i])
    return result

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


# This function checks whether a given wavefunction is normalized or not.
def SquareSum(some_psi):
    sum = 0
    for i in range(x.size):
        sum += np.abs(some_psi[i])**2
    return sum


# This function returns the excited state wavefunction for the specified order.
# order = 0 corresponds to the groundstate, order = 1 corresponds to the 1st
# excited state vice versa.
def ExcitedState(order):
    H = HMatrix()
    val,vec=np.linalg.eig(H)
    z = np.argsort(val)

    psi = vec[:,z[order]]
    return psi

# This function returns the coherent state wavefunction for the specified force
# constant (k). k is obtained by multiplying the kappa with a scaling factor.
def CoherentState(sf):
    psi = ExcitedState(0)
    k = kappa*sf
    psi = np.exp(1j*k*x)*psi

    return psi

# This is an implementation of Eqn. 14 in 
# https://www.overleaf.com/project/614c914c7242ebd307c3b49c
def AnalyticInitial():
    psi = np.zeros((x.size))
    for i in range(x.size):
        psi[i] = (np.pi**(-0.25))*np.exp(-0.5*(x[i]**2))

    return psi



# 𝔻𝔼𝕄𝕆 𝕊𝔼ℂ𝕋𝕀𝕆ℕ

numericalGroundstate = ExcitedState(0)
analyticalGroundstate = AnalyticInitial()

print("Numerical groundstate: " + str(SquareSum(numericalGroundstate)))
print("Analytical groundstate: " +
        str(SquareSum(analyticalGroundstate)*dx)) 
#Potential
plt.subplot(2, 1, 1)
plt.plot (x, np.abs(numericalGroundstate)**2, "r--", label=r"$|\Psi(x,t=0)|^2\;,\;numerical$")

plt.subplot(2, 1, 2)
plt.plot (x, np.abs(analyticalGroundstate)**2, "r--", label=r"$|\Psi(x,t=0)|^2\;,\;numerical$")

plt.show()




"""
fig = plt.figure(figsize=(16, 9), dpi=1920/16)
for i in range(timesteps):
    currentTime = i*dt
    plt.suptitle('t = {:.5f}'.format(currentTime) )

    #Potential
    plt.subplot(5, 1, 1)
    plt.plot (x, Potential(x), "r--", label=r"$V(x)$")

    # Coherent States with different force constants
    plt.subplot(5, 1, 2)
    k1 = 0.15*kappa
    plt.plot(x, np.abs(coh1)**2, "b-", label=r"$|\Psi_1(x,t)|^2\;,\;k=$"+'{:.4f}'.format(k1))
    plt.legend()

    plt.subplot(5, 1, 3)
    k2 = 0.05*kappa
    plt.plot(x, np.abs(coh2)**2, "b-", label=r"$|\Psi_2(x,t)|^2\;,\;k=$"+'{:.4f}'.format(k2))
    plt.legend()

    plt.subplot(5, 1, 4)
    k3 = 0.01*kappa
    plt.plot(x, np.abs(coh3)**2, "b-", label=r"$|\Psi_3(x,t)|^2\;,\;k=$"+'{:.4f}'.format(k3))
    plt.legend()

    plt.subplot(5, 1, 5)
    k4 = 0.005*kappa
    plt.plot(x, np.abs(coh4)**2, "b-", label=r"$|\Psi_4(x,t)|^2\;,\;k=$"+'{:.4f}'.format(k4))
    plt.legend()



    plt.tight_layout()

    fig.savefig('coherent_test'+str(i)+'.png')
    plt.clf()

    coh1 = Evolve(coh1)
    coh2 = Evolve(coh2)
    coh3 = Evolve(coh3)
    coh4 = Evolve(coh4)
"""
