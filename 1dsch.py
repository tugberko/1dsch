#!/usr/bin/python
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np

hbar = 1
m = 1

dt = 1e-3
halfdt = dt*0.5

dx    = 2*1e-2                # spatial separation
x     = np.arange(-25, 25, dx)       # spatial grid points

T=1
omega = 2 * np.pi / T

def IdentityMatrix():
    I = np.zeros((x.size, x.size))
    for i in range(x.size):
        I[i][i] = 1
    return I

def Potential(pos):
    return 0.5*m*(omega**2)*(pos**2)

def TMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size):
        result[i][i]=-2
    for i in range(x.size-1):
        result[i][i+1]=1
        result[i+1][i]=1
    return ( -(hbar**2)/(2*m) ) * (1/dx**2) * result


def VMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size):
        result[i][i]=Potential(x[i])
    return result

def HMatrix():
    return TMatrix() + VMatrix()

def AnnihilationMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size-1):
        result[i][i+1] = np.sqrt(i+1)
    return result

def CreationMatrix():
    result = np.zeros((x.size,x.size))
    for i in range(x.size-1):
        result[i+1][i] = np.sqrt(i+1)
    return result

def LHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2*hbar)

    return first_term + second_term

def RHS():
    first_term = IdentityMatrix()
    second_term = (1j*HMatrix()*dt)/(2*hbar)

    return first_term - second_term

LHS = LHS()
RHS = RHS()
InverseOfLHS = np.linalg.inv(LHS)
U = np.matmul(InverseOfLHS, RHS)

def Evolve(some_psi):
    return np.matmul(U, some_psi)


def Gaussian():
    kx    = 0.1                        # wave number
    sigma = 0.25                        # width of initial gaussian wave-packet
    x0    = 3.0                        # center of initial gaussian wave-packet

    A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

    # Initial Wavefunction
    return np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)


def IsNormalized(some_psi):
    sum = 0
    for i in range(x.size):
        sum += np.abs(some_psi[i])**2
    return sum





def CoherentEvolution():
    A = AnnihilationMatrix()
    val,vec=np.linalg.eig(A)
    z = np.argsort(val)
    coh = vec[:,z[0]]

    #Evolution of coherent state
    timesteps = 3600
    fig = plt.figure(figsize=(12, 8))
    for i in range(timesteps):
        print("Drawing frame "+str(i)+" of "+str(timesteps))
        currentTime = i*dt
        #plt.plot (x, Potential(x)*0.01, "r--", label=r"$V(x)$")
        plt.plot(x, np.abs(coh)**2, "b-", label=r"$|\Psi(x,t)|^2$")
        #plt.ylim(0,0.005)
        #plt.xlim(-15,15)
        plt.xlabel("Position")
        normalization=IsNormalized(coh)
        plt.title('t = {:.4f}'.format(currentTime) + " s\n"+ 'normalization: {:.7f}'.format(normalization))
        plt.legend()
        fig.savefig('new_coherent'+str(i)+'.png')
        plt.clf()
        coh = Evolve(coh)


def GaussianEvolution():
    #Evolution of Gaussian wave packet
    psi = Gaussian()
    timesteps = 3000
    fig = plt.figure(figsize=(12, 8))
    for i in range(timesteps):
        currentTime = i*dt
        plt.plot (x, Potential(x)*0.01, "r--")
        plt.plot(x, np.abs(psi)**2, "b-", label=r"$|\Psi_0(x,t)|^2$")
        plt.ylim(0,5)
        plt.xlim(-5,5)
        plt.xlabel("Position")
        normalization=IsNormalized(psi)
        plt.title('t = {:.4f}'.format(currentTime) + " s\n"+ 'normalization: {:.7f}'.format(normalization))
        fig.savefig('gaussian'+str(i)+'.png')
        plt.clf()
        psi = Evolve(psi)



def GroundState():
    H = HMatrix()
    val,vec=np.linalg.eig(H)
    z = np.argsort(val)
    z = z[0:4]
    energies=(val[z]/val[z][0])
    print(energies)


    psi0 = vec[:,z[0]]
    #psi1 = vec[:,z[1]]
    
    #Evolution of groundstate
    timesteps = 500
    fig = plt.figure(figsize=(12, 8))
    for i in range(timesteps):
        currentTime = i*dt
        plt.plot (x, Potential(x)*0.01, "r--", label=r"$V(x)$")
        plt.plot(x, np.abs(psi0)**2, "b-", label=r"$|\Psi_0(x,t)|^2$")
        plt.ylim(0,0.1)
        plt.xlim(-5,5)
        plt.xlabel("Position")
        normalization=IsNormalized(psi0)
        plt.title('t = {:.4f}'.format(currentTime) + " s\n"+ 'normalization: {:.7f}'.format(normalization))
        plt.legend()
        fig.savefig('groundstate'+str(i)+'.png')
        plt.clf()
        psi0 = Evolve(psi0)



def FirstExcited():
    H = HMatrix()
    val,vec=np.linalg.eig(H)
    z = np.argsort(val)
    z = z[0:4]
    energies=(val[z]/val[z][0])
    print(energies)
    
    #psi0 = vec[:,z[0]]
    psi1 = vec[:,z[1]]
    
    #Evolution of first excited state
    timesteps = 500
    fig = plt.figure(figsize=(12, 8))
    for i in range(timesteps):
        currentTime = i*dt
        plt.plot (x, Potential(x)*0.01, "r--", label=r"$V(x)$")
        plt.plot(x, np.abs(psi1)**2, "b-", label=r"$|\Psi_1(x,t)|^2$")
        plt.ylim(0,0.1)
        plt.xlim(-5,5)
        plt.xlabel("Position")
        normalization=IsNormalized(psi1)
        plt.title('t = {:.4f}'.format(currentTime) + " s\n"+ 'normalization: {:.7f}'.format(normalization))
        plt.legend()
        fig.savefig('firstexcited'+str(i)+'.png')
        plt.clf()
        psi1 = Evolve(psi1)



