import numpy as np
from DAE import DAE
import random


def run_dae():
    drug_train = np.loadtxt(r"../feature/drug_vector.txt")
    protein_train = np.loadtxt(r"../feature/protein_vector.txt")

    drug_size=drug_train.shape[1]
    protein_size=protein_train.shape[1]

    print(drug_size,protein_size)
    drug_feature=DAE(drug_train,drug_size,20,16,1,100,[100])
    np.savetxt('../feature/drug_dae_d100.txt',drug_feature)

    protein_feature=DAE(protein_train,protein_size,20,32,1,400,[400])
    np.savetxt('../feature/protein_dae_d400.txt',protein_feature)
    return drug_feature, protein_feature

def generate_pair():
    interaction = np.loadtxt('../data/mat_drug_protein.txt')
    drug = np.loadtxt('../feature/drug_dae_d100.txt')
    protein = np.loadtxt('../feature/protein_dae_d400.txt')
    pos_index = np.where(interaction == 1)
    allneg_index = np.where(interaction == 0)

    posx = pos_index[0]
    posy = pos_index[1]

    allnegx = allneg_index[0]
    allnegy = allneg_index[1]
    allneg = np.transpose(np.vstack((allnegx, allnegy)))

    result = random.sample(range(0, len(allneg)), len(posx))
    index = np.random.randint(len(allneg), size=len(posx))
    neg = allneg[list(result)]
    negx = neg[:, 0]
    negy = neg[:, 1]

    p1 = []
    n1 = []
    for i, j in zip(posx, posy):
        dv = drug[i]
        pv = protein[j]
        p1.append(np.hstack((dv, pv)))
    # print(np.shape(p1))

    for m, n in zip(negx, negy):
        dnv = drug[m]
        pnv = protein[n]
        n1.append(np.hstack((dnv, pnv)))
    # print(np.shape(n1))

    pos_label = np.ones((len(posx), 1))
    neg_label = np.zeros((len(posx), 1))
    data = np.vstack((p1, n1))
    # print(np.shape(data))
    label = np.vstack((pos_label, neg_label))
    np.savetxt('../feature/data.txt', data)
    # np.savetxt('../cnn_input/data1.txt',posx)
    np.savetxt('../feature/label.txt', label)

if __name__ == "__main__":
    drug_feature, protein_feature = run_dae()
    generate_pair()
