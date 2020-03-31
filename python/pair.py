import numpy as np
import random
import pandas as pd
from itertools import chain
interaction = np.loadtxt('../data/mat_drug_protein.txt')
drug=np.loadtxt('../DAE/drug_dae_d100.txt')
protein=np.loadtxt('../DAE/protein_dae_d400.txt')

pos_index=np.where(interaction==1)
allneg_index=np.where(interaction==0)

posx = pos_index[0]
posy = pos_index[1]
print(posx)
print(posy)

allnegx = allneg_index[0]
allnegy = allneg_index[1]
allneg=np.transpose(np.vstack((allnegx,allnegy)))

result=random.sample(range(0,len(allneg)),len(posx))
# print(result)
# print(len(result))
index = np.random.randint(len(allneg), size=len(posx))
neg = allneg[list(result)]
negx=neg[:,0]
negy=neg[:,1]

print(len(negx))

p1=[]
n1=[]
# print(drug)
for i,j in zip(posx,posy):
    # print(k)
    dv=drug[i]
    pv=protein[j]
    # print(dv)
    p1.append(np.hstack((dv,pv)))
    # pos_vector=np.append(dv,pv)
# print(p1)
print(np.shape(p1))
#
for m,n in zip(negx,negy):
    dnv=drug[m]
    pnv=protein[n]
    n1.append(np.hstack((dnv,pnv)))
# print(p1)
print(np.shape(n1))

pos_label=np.ones((len(posx), 1))
neg_label=np.zeros((len(posx), 1))
data=np.vstack((p1,n1))
print(np.shape(data))
label=np.vstack((pos_label,neg_label))
np.savetxt('../cnn_input/data.txt',data)
np.savetxt('../cnn_input/data1.txt',posx)

np.savetxt('../cnn_input/label.txt',label)