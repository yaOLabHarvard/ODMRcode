import numpy as np
from math import radians

def effectivePos(theta, phi):
    ## Given fitted theta and phi from the ESR spectra, obtain all possible
    ## orientations in [100]
    ## both arguments are in radians
    allPos = []
    allTheta = [theta, -theta, np.pi+theta, np.pi-theta]
    allPhi = [phi, -phi, np.pi/2-phi, np.pi/2+phi, np.pi-phi, np.pi+phi, \
              3*np.pi/2-phi, 3*np.pi/2+phi]
    for t in allTheta:
        for p in allPhi:
            allPos.append([t, p])

    return np.array(allPos)


def fieldSum(B1, pos1, B2, pos2):
    ## Given all possible orientations for field 1 and 2: pos1 and pos2
    ## sum them up to get all possible results

    sumResult = []

    i = 0
    j = 0
    for b1 in pos1:
        for b2 in pos2:
            bx = B1*np.sin(b1[0])*np.cos(b1[1])+B2*np.sin(b2[0])*np.cos(b2[1])
            by = B1*np.sin(b1[0])*np.sin(b1[1])+B2*np.sin(b2[0])*np.sin(b2[1])
            bz = B1*np.cos(b1[0])+B2*np.cos(b2[0])
            sumResult.append([i, j, bx,by,bz])
            j += 1
        i += 1

    return np.array(sumResult)

def compareDist(sumR, pos3):
    ## Given all the possible sum result sumR and one possible direction of
    ## sum field pos3, find the right direction of pos1 and pos2

    ## change polar to xyz
    pos3 = np.array([pos3[0]*np.sin(pos3[1])*np.cos(pos3[2]),\
                     pos3[0]*np.sin(pos3[1])*np.sin(pos3[2]),\
                     pos3[0]*np.cos(pos3[1])])

    dist = np.zeros((len(sumR), 3))
    i = 0
    for pos in sumR:
        d = np.linalg.norm(pos3 - pos[-3:])
        dist[i][0] = d
        dist[i][1] = pos[0]
        dist[i][2] = pos[1]
        i += 1
    dist = dist[dist[:, 0].argsort()]
    print(dist[:20])

def getRatio(b1a,b1t,b1p,b2a, b2t,b2p):
    b1 = np.array([b1a*np.sin(b1t)*np.cos(b1p), b1a*np.sin(b1t)*np.sin(b1p), b1a*np.cos(b1t)])
    b2 = np.array([b2a*np.sin(b2t)*np.cos(b2p), b2a*np.sin(b2t)*np.sin(b2p), b2a*np.cos(b2t)])
    ratio = np.arange(0,10,.1)
    theta = np.zeros(len(ratio))
    i = 0
    for r in ratio:
        newb = b1*r + b2
        newtheta = np.arctan(np.sqrt(newb[0]**2 +newb[1]**2)/newb[2])/np.pi*180
        theta[i] = newtheta
        i += 1
    theta = np.abs(theta -54.74)
    return np.argmin(theta)*.1

def Ratiosum(b1a,b1t,b1p,b2a, b2t,b2p, r, hh):
    b1 = np.array([b1a*np.sin(b1t)*np.cos(b1p), b1a*np.sin(b1t)*np.sin(b1p), b1a*np.cos(b1t)])
    b2 = np.array([b2a*np.sin(b2t)*np.cos(b2p), b2a*np.sin(b2t)*np.sin(b2p), b2a*np.cos(b2t)])

    return hh*(r*b1 + b2)

B1alpha = 10.24
B1theta = radians(75.66)
B1phi = radians(10.32)
B2alpha = 32.67
B2theta = radians(7.25)
B2phi = radians(20.37)
B3alpha = 34.9
B3theta = radians(11.8)
B3phi = radians(34.4)

##B1alpha = 10
##B1theta = 0
##B1phi = 0
##B2alpha = 10
##B2theta = radians(90)
##B2phi = radians(90)
##B3alpha = 10*np.sqrt(2)
##B3theta = radians(45)
##B3phi = 0


p1 = effectivePos(B1theta, B1phi)
p2 = effectivePos(B2theta, B2phi)
print(p1)
print(p2)
##print("\n")
p3 = np.array([B3alpha, B3theta, B3phi])
fsum = fieldSum(B1alpha,p1,B2alpha,p2)
#print(len(fsum))
compareDist(fsum, p3)

B2theta = p2[4][0]
B2phi = p2[4][1]
print(B2phi)

print(getRatio(B1alpha,B1theta,B1phi, B2alpha,B2theta,B2phi))
print(Ratiosum(B1alpha,B1theta,B1phi, B2alpha,B2theta,B2phi, 7.8, 0.5))

    
