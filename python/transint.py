import numpy as np

def trans(f,hang,lie):
    line = f.readline()
    hang=int(hang)
    lie=int(lie)
    data = np.zeros((hang,lie))
    i = 0
    while line:
     num = np.array([float(x) for x in line.split()])
     data[i,:] = num
     line = f.readline()
     i = i+1
    return(data)