import os, os.path
import transint
import numpy as np

trainPath = '../cnn_input/train/'
testPath = '../cnn_input/test/'
trainlabelPath = '../cnn_input/trainlabel/'
testlabelPath = '../cnn_input/testlabel/'

rownum=500
txtType = 'txt'
txtLists = os.listdir(trainPath)  # 列出文件夹下所有的目录与文件
txtLists.sort()
trainrow=np.loadtxt('../cnn_input/trainrow.txt')
testrow=np.loadtxt('../cnn_input/trainrow.txt')


def printall(path,row,judge):
    index = 0
    txtLists = os.listdir(path)
    txtLists.sort()
    allint=[1,2,3,4,5,6,7,8,9,10]
    for filename in txtLists:
        f = open(path + filename)
        a = transint.trans(f,row[index], judge)
        allint[index]=a
        index=index+1
    return(allint)


trainall=printall(trainPath,trainrow,rownum)
testall=printall(testPath,testrow,rownum)

trlabelall=printall(trainlabelPath,trainrow,1)
telabelall=printall(testlabelPath,testrow,1)

