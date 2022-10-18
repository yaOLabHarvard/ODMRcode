from os.path import isfile
from os.path import join
from os import listdir
import os
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.spatial.transform import Rotation as Rot
import numpy as np

import matplotlib.ticker as ticker
from matplotlib import gridspec
from scipy.io import loadmat

curve_style = 'l'

## rotation matrix
def RMat(Rangle, Raxis):
    # Rangle is the rotation angle in radian; Raxis is the normalized rotation axis
    r = Rot.from_rotvec(Rangle* np.array(Raxis)/np.linalg.norm(Raxis))
    return r.as_matrix()

## spin 1 operators
Sx= np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
Sy= np.array([[0,1,0],[-1,0,1],[0,-1,0]])/(np.sqrt(2)*1j)
Sz= np.array([[1,0,0],[0,0,0],[0,0,-1]])


# peak fit functions
def gaussian(xVals,A,sigma,x0, eta = 0):
    '''
    Returns a gaussian for the fit function
    '''
    return A*np.exp(-(xVals-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

# lorentzian fit functions
def lorentzian(xVals,A,gamma,x0, eta = 0):
    '''
    Returns a gaussian for the fit function
    '''
    return A/np.pi*((gamma/2.)/((xVals-x0)**2+(gamma/2.)**2))

# linear combination of gaussian and lorentzian
def PseudoVoigt(xVals,A, sigma, x0, eta=0.5):
    # eta between 0 and 1
    return eta*gaussian(xVals,A,sigma,x0) + (1 - eta)*lorentzian(xVals,A,np.sqrt(2*np.log(2))*sigma,x0)

funcMap = {'l':lorentzian, 'g':gaussian, 'v':PseudoVoigt}


class NVfit:

    def __init__(self):
        
        #Gyromagnetic ratio
        self.gamma = 28025*100 #Hz/Guass
        #Total 11 fitting parameters
        self.PGuess = np.zeros(11)
        #zero field splitting
        self.PGuess[0] = 2.870e9 #Hz
        #Electric field
        self.PGuess[1] = 0
        #Magnetic field intensity
        self.PGuess[2] = 67 #G
        self.PGuess[3] = 30 #in-plane angle in deg
        self.PGuess[4] = 20 # out of plane angle in deg
        self.PGuess[5] = 10e6 # Hz
        self.PGuess[6] = 52.75
        self.PGuess[7] = -20e6
        self.PGuess[8] = -20e6
        self.PGuess[9] = -10e6
        self.PGuess[10] = -10e6 # Intensity offset and individual intensity for four NV groups
        
        
        print("NV initialized!")


    def NV100_exact_Bonly(self):
        ## Due to symmetry, the only parameter matters is the angle between NV axis and magnetic field. This is true
        ## when all the terms in the Hamilitonian are axial

            
        ## define magnetic field direction
        theta = self.PGuess[4]*np.pi/180
        phi = self.PGuess[3]*np.pi/180
        Bdir = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        ## define angle between B and NV axis
        alpha=np.zeros(4)
        alpha[0] = np.arccos(Bdir.dot([1,1,1])/np.sqrt(3))
        alpha[1] = np.arccos(Bdir.dot([-1,-1,1])/np.sqrt(3))
        alpha[2] = np.arccos(Bdir.dot([-1,1,-1])/np.sqrt(3))
        alpha[3] = np.arccos(Bdir.dot([1,-1,-1])/np.sqrt(3))
        #print(alpha)
        ## Calculate frequency
        MWFreq = np.zeros(8)
        for i in range(4):
            HH = self.PGuess[0]*Sz.dot(Sz)+self.gamma*self.PGuess[2]*(np.cos(alpha[i])*Sz+np.sin(alpha[i])*Sx)
            eigval = np.sort(np.real(np.linalg.eigvals(HH)))
            MWFreq[2*i] = eigval[1] - eigval[0]
            MWFreq[2*i + 1] = eigval[2] - eigval[0]

        return MWFreq


    def Plot_guess_fit(self, xVals, MWFreq, style = 'l'):
        eta = 0.5
        function = np.ones(len(xVals))*self.PGuess[6]

        for i in range(8):
            num = 7 + int(i/2)
            function += funcMap[style](xVals, self.PGuess[num], self.PGuess[5], MWFreq[i], eta)
        #print(function)
        return  function

NVFit = NVfit()
    
        
def NV100_fitting_function(xVals, DD, EE, BB, phi, theta, width, offset, A1, A2, A3, A4):
    
    NVFit.PGuess[0] = DD
    NVFit.PGuess[1] = EE
    NVFit.PGuess[2] = BB
    NVFit.PGuess[3] = phi
    NVFit.PGuess[4] = theta
    NVFit.PGuess[5] = width
    NVFit.PGuess[6] = offset
    NVFit.PGuess[7] = A1
    NVFit.PGuess[8] = A2
    NVFit.PGuess[9] = A3
    NVFit.PGuess[10] = A4
    print(A1)
    MWFreq = NVFit.NV100_exact_Bonly()

    return NVFit.Plot_guess_fit(xVals, MWFreq, curve_style)
    
            

