import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models
from keras import layers
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
import keras
import yaml

CONFIG_PATH = "./config/"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config("Models_utilitiesconfig.yaml")

plt.style.use('ggplot')


#
def set_seed(s):
    np.random.seed(s)
    random.seed(s)
    tf.random.set_seed(s)


# imbalanced data upsampling
# SMOTE
def Upsampling_SMOTE(X, y):  # The X is traning dataset, y is training labels
    smo = SMOTE(random_state=0)
    print("The original distribution:", Counter(y))
    X_smo, y_smo = smo.fit_resample(X, y)
    print("The upsampling distribution:", Counter(y_smo))
    return X_smo, y_smo


# split according to cancer type
def CancerType_Split(Xn, Y, Yb, Xnheader):
    Xn = pd.DataFrame(Xn, columns=Xnheader)
    Y = pd.DataFrame(Y, columns=["Tumor type"])
    Yb = pd.DataFrame(Yb, columns=["C0", "C1", "C2", "C3", "C4", "C5", "C6"])

    df = pd.concat([Xn, Y, Yb], axis=1)
    # ['Breast', 'Colorectum', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Upper GI']
    C0 = df[df["Tumor type"] == 0]
    C1 = df[df["Tumor type"] == 1]
    C2 = df[df["Tumor type"] == 2]
    C3 = df[df["Tumor type"] == 3]
    C4 = df[df["Tumor type"] == 4]
    C5 = df[df["Tumor type"] == 5]
    C6 = df[df["Tumor type"] == 6]
    return C0, C1, C2, C3, C4, C5, C6


# model
# Pre-train model
### model_0 2-layer
def Pretrain_model(X, Yb, f):
    # MLP
    Ptnn = models.Sequential()
    Ptnn.add(layers.Dense(config['layer1_node'], activation='relu', input_shape=(40,)))
    Ptnn.add(layers.Dense(config['layer2_node'], activation='relu'))
    Ptnn.add(layers.Dense(2, activation='sigmoid'))  # multi-class softmax
    '''
    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='loss',  # use validation accuracy for stopping
        min_delta=0.0001, patience=3,
        verbose=1)
    callbacks_list = [earlystop]

    Ptnn.compile(loss=keras.losses.binary_crossentropy,
               optimizer=keras.optimizers.SGD(lr=0.00001, momentum=0.9),
               metrics=['accuracy']  # also calculate accuracy during training
               )
    Ptnn.fit(X, Yb, epochs=50, batch_size=16,
           callbacks=callbacks_list,
           verbose=True)
    '''
    Ptnn.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    Ptnn.fit(X, Yb, epochs=config['pretrain_epochs'], batch_size=config['pretrain_batchsizes'], verbose=1)

    Ptnn.save(f)

### model_1 1-layer
def Pretrain_model_1(X, Yb, f):
    # MLP
    Ptnn = models.Sequential()
    Ptnn.add(layers.Dense(64, activation='relu', input_shape=(40,)))
    Ptnn.add(layers.Dense(2, activation='sigmoid'))  # multi-class softmax
    Ptnn.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    Ptnn.fit(X, Yb, epochs=10, batch_size=10, verbose=0)
    Ptnn.save(f)

### model_2 3-layer
def Pretrain_model_2(X, Yb, f):
    # MLP
    Ptnn = models.Sequential()
    Ptnn.add(layers.Dense(128, activation='relu', input_shape=(40,)))
    Ptnn.add(layers.Dense(32, activation='relu'))
    Ptnn.add(layers.Dense(64, activation='relu'))
    Ptnn.add(layers.Dense(2, activation='sigmoid'))  # multi-class softmax

    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='loss',  # use validation accuracy for stopping
        min_delta=0.0001, patience=3,
        verbose=1)
    callbacks_list = [earlystop]

    Ptnn.compile(loss=keras.losses.binary_crossentropy,
                 optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                 metrics=['accuracy']  # also calculate accuracy during training
                 )
    Ptnn.fit(X, Yb, epochs=50, batch_size=16,
             callbacks=callbacks_list,
             verbose=True)

    '''
    Ptnn.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    Ptnn.fit(X, Yb, epochs=50, batch_size=10, verbose=1) ###
    '''
    Ptnn.save(f)


### model_1 4-layer
def Pretrain_model_3(X, Yb, f):
    # MLP
    Ptnn = models.Sequential()
    Ptnn.add(layers.Dense(64, activation='relu', input_shape=(40,)))
    Ptnn.add(layers.Dense(64, activation='relu'))
    Ptnn.add(layers.Dense(64, activation='relu'))
    Ptnn.add(layers.Dense(64, activation='relu'))
    Ptnn.add(layers.Dense(2, activation='sigmoid'))  # multi-class softmax
    Ptnn.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    Ptnn.fit(X, Yb, epochs=10, batch_size=10, verbose=0)
    Ptnn.save(f)

# Freeze_layers
def Freeze_Layer(filename):
    Ptnn = load_model(filename)
    M2 = models.Sequential()
    for layer in Ptnn.layers[:-1]:  # go through until last layer
        M2.add(layer)
    for layer in M2.layers:
        layer.trainable = False
        # print(M2.summary())
    return M2


# Fine-tune model
### Fine-tune model_0
def Finetune_model(X, Yb, M2):
    # MLP
    M2.add(layers.Dense(config['layera_node'], activation='relu', name="dense_a"))
    M2.add(layers.Dense(config['layerb_node'], activation='relu', name="dense_b"))
    M2.add(layers.Dense(config['layerc_node'], activation='relu', name="dense_c"))
    M2.add(layers.Dense(2, activation='sigmoid', name="dense_output"))

    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='loss',  # use validation accuracy for stopping
        min_delta=0.001, patience=3,
        verbose=1)
    callbacks_list = [earlystop]

    M2.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy']) # default lr=0.001
    M2.fit(X, Yb, epochs=config['target_epochs'], batch_size=config['target_batchsizes'], callbacks=callbacks_list, verbose=1)
    # print(M2.summary())
    return M2

def Finetune_model_lr1e5(X, Yb, M2):
    # MLP
    M2.add(layers.Dense(64, activation='relu', name="dense_a"))
    M2.add(layers.Dense(64, activation='relu', name="dense_b"))
    M2.add(layers.Dense(64, activation='relu', name="dense_c"))
    M2.add(layers.Dense(2, activation='sigmoid', name="dense_output"))

    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='loss',  # use validation accuracy for stopping
        min_delta=0.0001, patience=5,
        verbose=1)
    callbacks_list = [earlystop]

    M2.compile(loss=keras.losses.binary_crossentropy,
               optimizer=keras.optimizers.SGD(lr=0.00001, momentum=0.9),
               metrics=['accuracy']  # also calculate accuracy during training
               )
    M2.fit(X, Yb, epochs=50, batch_size=16,
                     callbacks=callbacks_list,
                     verbose=True)
    '''
    # M2.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy']) # default lr=0.001
    myoptimizer = keras.optimizers.RMSprop(lr=0.00001)
    M2.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])
    M2.fit(X, Yb, epochs=10, batch_size=10, verbose=0)
    # print(M2.summary())
    '''
    return M2

### Fine-tune model_1
def Finetune_model_1(X, Yb, M2):
    # MLP
    M2.add(layers.Dense(64, activation='relu', name="dense_a"))
    M2.add(layers.Dense(2, activation='sigmoid', name="dense_output"))

    M2.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    M2.fit(X, Yb, epochs=10, batch_size=10, verbose=0)
    # print(M2.summary())

    return M2

def Finetune_model_1_lr1e5(X, Yb, M2):
    # MLP
    M2.add(layers.Dense(64, activation='relu', name="dense_a"))
    M2.add(layers.Dense(2, activation='sigmoid', name="dense_output"))


    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='loss',  # use validation accuracy for stopping
        min_delta=0.0001, patience=3,
        verbose=1)
    callbacks_list = [earlystop]

    M2.compile(loss=keras.losses.binary_crossentropy,
               optimizer=keras.optimizers.SGD(lr=0.00001, momentum=0.9),
               metrics=['accuracy']  # also calculate accuracy during training
               )
    M2.fit(X, Yb, epochs=50, batch_size=16,
                     callbacks=callbacks_list,
                     verbose=True)
    '''

    myoptimizer = keras.optimizers.RMSprop(lr=0.00001)
    M2.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])
    M2.fit(X, Yb, epochs=50, batch_size=10, verbose=1)
    # print(M2.summary())
    '''
    return M2


### Fine-tune model_2  2layer
def Finetune_model_2(X, Yb, M2):
    # MLP
    M2.add(layers.Dense(64, activation='relu', name="dense_a"))
    M2.add(layers.Dense(64, activation='relu', name="dense_b"))
    M2.add(layers.Dense(2, activation='sigmoid', name="dense_output"))

    M2.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    M2.fit(X, Yb, epochs=10, batch_size=10, verbose=1)
    # print(M2.summary())

    return M2

# CancerType_Selection
'''
def CancerTypeCombination_generator(Grouplist):
    combinator = []
    for source in Grouplist:
        for target in Grouplist:
            if(source != target):
                combination = []
                combination.append(source)
                combination.append(target)
                combinator.append(combination)
    return combinator
'''


def CancerTypeCombination_generator(*a, k=None):  # permutation
    '''Returns the list of (k-)permutations of the position arguments.'''
    n = len(a)
    output = []
    if k is None:
        k = n
    if 0 < k <= n:
        for i in range(n):
            output.extend([(a[i],) + p for p in CancerTypeCombination_generator(*a[:i], *a[i + 1:], k=k - 1)])
    elif k == 0:
        return [()]
    return output


def deduplicate(combinator):
    output = []
    l = len(combinator)
    index = np.ones((l))
    for i in range(l):
        for j in range(i + 1, l):
            if (list(set(combinator[i]).difference(set(combinator[j]))) == []):
                index[j] = 0

    for i in range(l):
        if (index[i] == 1):
            output.append(combinator[i])
    return output



def get_dataset_from_CancerTypeName(Xn, Y, Yb, Xcolname, CancerTypeName):
    C0, C1, C2, C3, C4, C5, C6 = CancerType_Split(Xn, Y, Yb, Xcolname)
    if CancerTypeName == 'C0':
        return C0
    if CancerTypeName == 'C1':
        return C1
    if CancerTypeName == 'C2':
        return C2
    if CancerTypeName == 'C3':
        return C3
    if CancerTypeName == 'C4':
        return C4
    if CancerTypeName == 'C5':
        return C5
    if CancerTypeName == 'C6':
        return C6


def Extract_DataGroup_1v1(Xn, Y, Yb, Xcolname, combination):  # e.g. combination = ["C0","C1"]
    CancerType0 = get_dataset_from_CancerTypeName(Xn, Y, Yb, Xcolname, combination[0])
    CancerType1 = get_dataset_from_CancerTypeName(Xn, Y, Yb, Xcolname, combination[1])
    DataGroup = CancerType0.append(CancerType1)
    DataGroup_X = DataGroup.iloc[:, :40]  # features
    DataGroup_Yb = DataGroup.loc[:, combination]  # onehot label
    return DataGroup_X, DataGroup_Yb


def Extract_DataGroup_1v5(Xn, Y, Yb, Xcolname,
                          combination):  # e.g. combination=["C0", ["C2","C3","C4","C5","C6"]] ##0 和 1位置应该对调
    CancerType0 = get_dataset_from_CancerTypeName(Xn, Y, Yb, Xcolname, combination[0])
    CancerType1 = []
    for Cn in combination[1]:
        CancerType1.append(get_dataset_from_CancerTypeName(Xn, Y, Yb, Xcolname, Cn))
    DataGroup = CancerType0.append(CancerType1)
    DataGroup_X = DataGroup.iloc[:, :40]  # features

    DataGroup_Yb1 = DataGroup.loc[:, combination[0]]  # onehot label
    temp = DataGroup_Yb1.copy()
    for Yn in range(len(DataGroup_Yb1)):
        temp.iloc[Yn] = bool(1 - DataGroup_Yb1.iloc[Yn]) + 0
    DataGroup_Yb = pd.concat([DataGroup_Yb1, temp], axis=1)
    DataGroup_Yb.columns = [combination[0], 'Cr']  # Cr contain 5 types of cancer
    return DataGroup_X, DataGroup_Yb

import pandas as pd
# Evaluation
def Auc_Ap_F1_Mcc_Acc(model, testX, testY, testYb):
    if (len(np.array(testYb)[0]) == 2):  # binary class
        pred = model.predict(testX)
        pred_class = model.predict_classes(testX)
        auc = roc_auc_score(testYb, pred)
        ap = average_precision_score(testYb, pred)
        f1 = f1_score(testY, pred_class)
        mcc = matthews_corrcoef(testY, pred_class)
        acc = accuracy_score(testY, pred_class)
    else:  # multiclass
        pred = model.predict(testX)
        pred_class = model.predict_classes(testX)
        auc = roc_auc_score(testYb, pred, multi_class="ovo", average='micro')
        ap = average_precision_score(testYb, pred)
        f1 = f1_score(testY, pred_class, average='micro')  ###
        mcc = matthews_corrcoef(testY, pred_class)
        acc = accuracy_score(testY, pred_class)
    return auc, ap, f1, mcc, acc


def Auc_Ap_F1_Mcc_Acc_Evaluation(pred, pred_class, testY, testYb):
    auc = roc_auc_score(testYb, pred, multi_class="ovo", average='micro')
    ap = average_precision_score(testYb, pred, average="micro")
    f1 = f1_score(testY, pred_class, average='micro')  ###
    mcc = matthews_corrcoef(testY, pred_class)
    acc = accuracy_score(testY, pred_class)
    return auc, ap, f1, mcc, acc

def Confusion_Matrix(y_test, predY):
    mat = confusion_matrix(y_test, predY)
    print(mat)


def Plot_BinaryClass_AUC(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.show()


def Plot_MultipleClass_AUC(tagnames, Yclasses, Yscores):
    fprall = []
    tprall = []
    aucall = []
    for i in range(len(tagnames)):
        fpr, tpr, thresholds = metrics.roc_curve(Yclasses[:, i], Yscores[:, i])
        plt.plot(fpr, tpr, lw=0.5, alpha=0.5)
        auc = metrics.auc(fpr, tpr)
        fprall.append(fpr)
        tprall.append(tpr)
        aucall.append(auc)

    # Then interpolate all ROC curves at this points
    all_fpr = np.unique(np.concatenate(fprall))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tagnames)):
        mean_tpr += np.interp(all_fpr, fprall[i], tprall[i])

    # Finally average it and compute AUC
    mean_tpr /= len(tagnames)

    # auc of the average ROC curve
    auc = metrics.auc(all_fpr, mean_tpr)

    # average AUC
    mc_auc = np.mean(aucall)

    plt.plot(all_fpr, mean_tpr, 'k-', lw=2)
    plt.title('MCAUC={:.4f}, AUC={:.4f}'.format(mc_auc, auc))

    plt.grid(True)

#Integration
def Pred_all_cancer_test(model,testX): # testsX including all cancertype
    Pred_all = model.predict(testX)
    return Pred_all

def Calculate_Cscores(models,PPAs,CancerType): #2D
    nsample = len(pd.DataFrame(PPAs[0])[0])
    df = pd.DataFrame(np.zeros(nsample))
    df_models = []
    for i in range(len(models)):
        if (models[i][1][0] == CancerType or models[i][1][1] == CancerType):  # model,target,position
            if (models[i][1][0] == CancerType):
                tempt = pd.DataFrame(PPAs[i])[0]
                df = pd.concat([df, tempt], axis=1)
                df_models.append(models[i])
            elif (models[i][1][1] == CancerType):
                tempt = pd.DataFrame(PPAs[i])[1]
                df = pd.concat([df, tempt], axis=1)
                df_models.append(models[i])
    return df, df_models

def Calculate_Cscores_1D(models,PPAs,CancerType): #2D
    nsample = len(pd.DataFrame(PPAs[0]))
    df = pd.DataFrame(np.zeros(nsample))
    df_models = []
    for i in range(len(models)):
        if (models[i][0] == CancerType or models[i][1] == CancerType):  # model,position
            if (models[i][0] == CancerType):
                tempt = pd.DataFrame(PPAs[i])[0]
                df = pd.concat([df, tempt], axis=1)
                df_models.append(models[i])
            elif (models[i][1] == CancerType):
                tempt = pd.DataFrame(PPAs[i])[1]
                df = pd.concat([df, tempt], axis=1)
                df_models.append(models[i])
    return df, df_models


#sorce_target name formation
def S_T_NameFormation(Cscore_model): #2D
    name = [Cscore_model[i][0][0]+Cscore_model[i][0][1]+"_"+Cscore_model[i][1][0]+Cscore_model[i][1][1] for i in range(len(Cscore_model))]#model, Source pr Target, First or Second
    return name

#sorce_target name formation
def S_T_NameFormation_1D(Cscore_model): #2D
    name = [Cscore_model[i][0]+Cscore_model[i][1] for i in range(len(Cscore_model))]#model, First or Second
    return name

#sorce_target name formation
def S_T_NameFormation_1vR(Cscore_model): #2D
    name = [Cscore_model[i][0][0]+Cscore_model[i][1][0] for i in range(len(Cscore_model))]#model, Source pr Target, First or Second
    return name

#sorce_target name formation
def S_T_NameFormation_1vR_1D(Cscore_model,Cancerlist): #1D
    name = []
    for i in range(len(Cscore_model)):
        P = list(set(Cancerlist).difference(Cscore_model[i][1]))
        P.remove(Cscore_model[i][0])
        myname = Cscore_model[i][0] + "Ex" + str(P[0])  # model, Source pr Target, First or Second
        name.append(myname)
    return name



