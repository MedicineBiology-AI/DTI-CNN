import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
import au_class as au
import pandas as pd

def standard_scale(X_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

def DAE(x_train,input_size,training_epochs,batch_size,display_step,lowsize,hidden_size):
    sdne = []
    ###initialize
    for i in range(len(hidden_size)):
        ae = au.Autoencoder(
            n_input=input_size,
            n_hidden=lowsize,
            transfer_function=tf.nn.softplus,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            scale=0.2)
        sdne.append(ae)
    Hidden_feature = []
    for j in range(len(hidden_size)):
        if j == 0:
            X_train = standard_scale(x_train)
        else:
            X_train_pre = X_train
            X_train = sdne[j - 1].transform(X_train_pre)
            Hidden_feature.append(X_train)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            for batch in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

                cost = sdne[j].partial_fit(batch_xs)
                # print("after = %f " % cost)

                avg_cost += cost / X_train.shape[0] * batch_size
            if epoch % display_step == 0:
                print("Epoch:", "%4d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        if j == 0:
            feat0 = sdne[0].transform(standard_scale(x_train))
            data1 = pd.DataFrame(feat0)
            print(data1.shape)
            np.set_printoptions(suppress=True)
    return data1

