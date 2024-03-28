import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def readFile(filename, deviceList = None, debug = 0):
    fr= open(path +filename+".txt", 'r')
    fs = open(path +filename+"_rho.txt", 'a+')
    fLines = fr.readlines()

    ##Aline = fLines[0::5]
    T1line = fLines[0::5]
    rho1line = fLines[1::5]
    rho2line = fLines[2::5]
    rho3line = fLines[3::5]
    T2line = fLines[4::5]

    if debug:
        ##print(Aline[0].split(' '))
        print(T1line[0].split(' '))
        print(rho1line[0].split('['))
        print(rho2line[0].split('['))
        print(rho3line[0].split('['))
        print(T2line[0].split(' '))

    rows = len(T1line)
    print("{} lines totoal".format(rows))

    if deviceList is not None:
        for i in range(len(deviceList)):
            fs.write("device name: {} \n".format(deviceList[i]))
            for j in range(rows):
                ##aa = float(Aline[j].split(' ')[-2])
                TT = (float(T1line[j].split(' ')[-2])+float(T2line[j].split(' ')[-2]))/2
                ##print(TT)
                ##print(j)
                r1l = rho1line[j].split('[')
                r2l = rho2line[j].split('[')
                r3l = rho3line[j].split('[')
                if r1l[0] == deviceList[i]:
                    rl = r1l
                elif r2l[0] == deviceList[i]:
                    rl = r2l
                elif r3l[0] == deviceList[i]:
                    rl = r3l
                else:
                    print("Name is wrong!!")
                    exit(0)
                
                ##print(rl)
                rl = rl[-1].split()
                ##print(rl)
                rx = float(rl[0])
                rxerr = float(rl[1])
                ry = float(rl[2])
                ryerr = float(rl[3].split(']')[0])
                    
                fs.write("{:.5e} {:.5e} {:.5e} {:.5e} {:.5e}\n".format(TT, rx, rxerr, ry, ryerr))
            fs.write("#################################\n")
    fr.close()
    fs.close()
    





path = "D:/work/Ni327/s1/transport/"
##filename = "ni327_s1_9p7gpa_warmup_0field"
filename = "ni327_s1_16p5gpa_warmup_0field_threelockin"

devicelist = ['mainSRS ', 'GPIB0::19::INSTR ', 'GPIB0::5::INSTR ']
output = readFile(filename, devicelist, debug=1)
