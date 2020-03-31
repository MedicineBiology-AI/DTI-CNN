import cnn1D
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import keras
from allfiles import trainall,testall,trlabelall,telabelall

roc=[]
pr=[]

for index in range(len(trainall)):
    print(index)
    # model[index] = cnn1D.build_model()
    model= keras.models.load_model('RunCnn.model')
    train = np.expand_dims(trainall[index], 2)
    test = np.expand_dims(testall[index],2)
    trainlabel = trlabelall[index]
    testlabel = telabelall[index]
    model.fit(train,trainlabel,
              batch_size=64, # 每个batch的大小为512
              epochs=35,  # 在全数据集上迭代20次
              )
    np.set_printoptions(precision=6)
    predict_y2 = model.predict_proba(test)
    # print(predict_y2)
    c=roc_auc_score(testlabel, predict_y2)
    precision, recall, thresholds = precision_recall_curve(testlabel, predict_y2)
    d = auc3(recall, precision)
    print("{:.4f} ".format(c)+"{:.4f} ".format(d))
    roc.append(c)
    pr.append(d)
print("roc:",roc)
print("pr：",pr)
print("auroc = {:.5f}".format(np.mean(roc)))
print("auprc = {:.5f}".format(np.mean(pr)))
