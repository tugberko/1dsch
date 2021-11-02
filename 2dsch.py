#!/usr/bin/python
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
import scipy.signal


# Simulation Parameters

dt = 8e-2   # Temporal seperation

# Spatial grid points
nx = 60
ny = 60

Lx = 10    # Spatial size, symmetric with respect to x=0
Ly = 10   # Spatial size, symmetric with respect to x=0

alpha=1
kx = Lx/200
ky = alpha * kx

# Derived Simulation Parameters

dx = Lx/(nx-1) # Spatial seperation
dy = Ly/(ny-1) # Spatial seperation

x = np.linspace(-0.5*Lx, 0.5*Lx, nx)
y = np.linspace(-0.5*Ly, 0.5*Ly, ny)


# This function takes size (n) and seperation (dx) as Parameters
# and produces a discrete 1D Laplacian matrix.
def laplacian1D(size, separation):
    result = np.zeros((size, size))
    for i in range(size):
        result[i][i]=-2
    for i in range(size-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return (-1/(2*(separation**2))) * result

# This function creates a discrete 2D Laplacian matrix by taking kronecker sum
# of two discrete 1D Laplacian matrix.
def laplacian2D():
    Dxx = laplacian1D(nx, dx)
    Dyy = laplacian1D(ny, dy)

    return sp.kron(sp.eye(ny), Dxx) + sp.kron(Dyy,sp.eye(nx))

# This function calculates the anisotropic potential energy at specified position.
# alpha is defined in such a way that   k_y = alpha * k_x   .
def Potential(posx, posy):
    return 0.5*(posx**2) + 0.5*(alpha * (posy**2))

# This function creates a discrete 2D potential energy matrix. It's compatible
# with flattening with row priority.
def VMatrix():
    result = np.zeros(( x.size*y.size , x.size*y.size ))
    cursor = 0
    for i in range(y.size):
        for j in range(x.size):
            result[cursor][cursor] = Potential(x[j] , y[i])
            cursor += 1

    return result

# This function creates a discrete 2D kinetic energy matrix.
def TMatrix():
    return -0.5 * laplacian2D().toarray()

# This function creates a discrete 2D Hamiltonian matrix.
def HMatrix():
    return VMatrix() + TMatrix()

def SquareSum2D(some_psi):
    sum = 0
    for i in range(len(some_psi)):
        for j in range(len(some_psi[i])):
            sum += some_psi[i][j] * np.conj(some_psi[i][j])

    return sum

def Normalize2D(some_psi):
    scale_factor = 1/SquareSum2D(some_psi)
    return scale_factor * some_psi


# This function finds the specified order of excited state with respect to
# given Hamiltonian. (order = 0 for groundstate, order = 1 for first excited
# state vice versa.)
def ExcitedState2d(order):
    H = HMatrix()
    val,vec=la.eig(H)
    z = np.argsort(val)

    psi = vec[:,z[order]]

    return Normalize2D(psi.reshape(ny, nx))


def IdentityMatrix():
    I = np.zeros((x.size*y.size, x.size*y.size))
    for i in range(x.size*y.size):
        I[i][i] = 1
    return I

def LHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2)

    return first_term + second_term

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
    return np.matmul(U, np.ndarray.flatten(some_psi)).reshape(ny,nx)


def InitialState():
    return ExcitedState2d(0) * np.exp(1j*kx*x) * np.exp(1j*ky*x)


timesteps = 100

psi = InitialState()

for i in range(timesteps):
    print('Timestep: ' + str(i))
    plt.contourf(x,y, np.abs(psi)**2,256, cmap='RdYlBu')
    plt.grid()
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('new' + str(i) +'.png')
    plt.clf()

    psi = Evolve(psi)
