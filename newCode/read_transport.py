import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def readFile(filename, issave= True):
    f= open(path +filename+".txt", 'r')
    fLines = f.readlines()

    startTline = fLines[0::3]
    rholine = fLines[1::3]
    endTline = fLines[2::3]

    print(startTline[0].split())
    print(rholine[2].split('['))
    print(endTline[0].split())

    rows = len(startTline)


    output = np.zeros((rows, 6), dtype = float)
    for i in range(rows):
        rl = rholine[i].split('[')[-1]
        rl = rl.split()

        staget1 = float(startTline[i].split()[2])
        staget2 = float(endTline[i].split()[2])

        samplet1 = float(startTline[i].split()[-2])
        samplet2 = float(endTline[i].split()[-2])
        
        if samplet1 == staget1:
            samplet1 = staget1*(1-0.143) + 20.791
            samplet2 = staget2*(1-0.143) + 20.791
            print(samplet1)
            print(samplet2)

        try:
            output[i, 0] = (samplet1+samplet2)/2
            output[i, 1] = (staget1+staget2)/2
            output[i, 2] = float(rl[0])
            output[i, 3] = float(rl[1])
            output[i, 4] = float(rl[2])
            output[i, 5] = float(rl[3][:-1])
        except:
            print(rl)
    if issave:
        np.savetxt(path + filename + '_rho.txt', output)

    return output

def tempDiff(rhoOutput, Npts = 1000, issave = True):
    rhoData = np.transpose(np.copy(rhoOutput))
    SampleT = rhoData[0]
    StageT = rhoData[1]
    Tdiff = SampleT - StageT
    print(Tdiff)
    TT = interpolate.interp1d(StageT, Tdiff)

    
    newX = np.linspace(StageT[0], StageT[-1], Npts)
    newDiff = TT(newX)
    
    plt.plot(StageT, Tdiff, 'o', newX, newDiff, '-')
    plt.show()
    plt.close()

    Tdiffsave = np.zeros((Npts, 2), dtype = float)
    for i in range(Npts):
        Tdiffsave[i,0] = newX[i]
        Tdiffsave[i,1] = newDiff[i]
    if issave:
        np.savetxt(path + filename + '_tdiff.txt', Tdiffsave)
    





path = "D:/work/LT2/"
##filename = "ni327_s1_9p7gpa_warmup_0field"
filename = "ni327_s1_16p5gpa_cooldown_0field"

output = readFile(filename, issave=True)
tempDiff(output)
