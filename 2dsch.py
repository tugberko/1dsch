#!/usr/bin/python
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
import datetime
import scipy.signal




# Simulation Parameters

dt = 1e-2   # Temporal seperation

nx = 40# Spatial grid points
ny = 40

Lx = 10    # Spatial size, symmetric with respect to x=0
Ly = 10    # Spatial size, symmetric with respect to x=0


# Derived Simulation Parameters

dx = Lx/(nx-1) # Spatial seperation
dy = Ly/(ny-1) # Spatial seperation

x = np.linspace(-0.5*Lx, 0.5*Lx, nx)
y = np.linspace(-0.5*Ly, 0.5*Ly, ny)


def laplacian1D(size, separation):
    result = np.zeros((size, size))
    for i in range(size):
        result[i][i]=-2
    for i in range(size-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return (-1/(2*(separation**2))) * result

def laplacian2D():
    Dxx = laplacian1D(nx, dx)
    Dyy = laplacian1D(ny, dy)

    return sp.kron(sp.eye(ny), Dxx) + sp.kron(Dyy,sp.eye(nx))

def Potential(posx, posy, alpha):
    return 0.5*(posx**2) + 0.5*(alpha * (posy**2))

def VMatrix():
    result = np.zeros(( x.size*y.size , x.size*y.size ))
    cursor = 0
    for i in range(x.size):
        for j in range(y.size):
            result[cursor][cursor] = Potential(x[i] , y[j], 64)
            cursor += 1

    return result

def TMatrix():
    return -0.5 * laplacian2D().toarray()


def HMatrix():
    return VMatrix() + TMatrix()


def ExcitedState2d(order):
    H = HMatrix()
    val,vec=la.eig(H)
    z = np.argsort(val)

    psi = vec[:,z[order]]

    return psi.reshape(ny, nx)


X, Y = np.meshgrid(x, y)

plt.contourf(X,Y, np.abs(ExcitedState2d(0))**2,256, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
