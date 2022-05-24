# %%
#revised in 20220112
#feature number from 40 change to 44
# 20 Sep, 2021
from Models_utilities import *
# from TL_model_v2_utilities import *
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import time
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import models
import yaml

# %%
CONFIG_PATH = "../03results/version6/config/"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("MLP_1v1config.yaml")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# set seed
os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(0)

def mkdir(path):
    folder = os.path.exists(path)
    pathcontrol = path + '/control'
    foldercontrol = os.path.exists(pathcontrol)

    # fold
    if not folder:
        os.mkdir(path)
        print("--- new folder... ---")
        print("--- OK ---")
    else:
        print("--- There is this folder! ---")

    # Controls
    if not foldercontrol:
        os.mkdir(pathcontrol)
        print("--- new folder... ---")
        print("--- OK ---")
    else:
        print("--- There is this folder! ---")

# %% save log
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# %%
#
def Mymodel_data_extraction(target, trainXn_UP, trainY_UP, trainYb_UP, testXn, testY, testYb, Xcolname):
    DataGroup_X, DataGroup_Yb = Extract_DataGroup_1v1(trainXn_UP, trainY_UP, trainYb_UP, Xcolname, target)
    testDataGroup2_X, DataGroup2_testYb = Extract_DataGroup_1v1(testXn, testY, testYb, Xcolname, target)
    return DataGroup_X, DataGroup_Yb, testDataGroup2_X, DataGroup2_testYb


def Mymodel_training(prefix, target, DataGroup_X, DataGroup_Yb):
    K.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(config['layer1_node'], activation='relu', input_shape=(41,)))
    model.add(layers.Dense(config['layer2_node'], activation='relu'))
    model.add(layers.Dense(config['layer3_node'], activation='relu'))
    model.add(layers.Dense(config['layer4_node'], activation='relu'))
    model.add(layers.Dense(config['layer5_node'], activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid', name="dense_output"))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
    model.fit(DataGroup_X, DataGroup_Yb, epochs=config['target_epochs'], batch_size=config['target_batchsizes'], verbose=0)

    model_filename = prefix + "/control/model" + target[0] + target[1] + "_Full_1v1.h5"
    model.save(model_filename)
    return model_filename

    # predict


def Mymodel_Predict(model, testDataGroup2_X, DataGroup2_testYb, testXn):
    DataGroup2_testY = [np.argmax(one_hot) for one_hot in np.array(DataGroup2_testYb)]
    auc, ap, f1, mcc, acc = Auc_Ap_F1_Mcc_Acc(model, testDataGroup2_X, DataGroup2_testY, DataGroup2_testYb)
    Pred_prob_alltest = Pred_all_cancer_test(model, testXn)  # testXn including all cancer types
    return auc, ap, f1, mcc, acc, Pred_prob_alltest


#
# %%
# ------------------------------------main----------------------------------
# data preprocess
Permutation = CancerTypeCombination_generator("C0", "C1", "C2", "C3", "C4", "C5", "C6", k=2)
L = len(Permutation)

# %%
def one_fold(prefix, trainXn, trainY, trainYb, testXn, testY, testYb, Xcolname):
    aucs = np.zeros(L)
    aps = np.zeros(L)
    f1s = np.zeros(L)
    mccs = np.zeros(L)
    accs = np.zeros(L)
    PPAs = []
    mymodels = []

    t0 = time.time()
    for i in range(L):
        target_combination = Permutation[i]  ######
        # data extraction
        DataGroup_X, DataGroup_Yb, testDataGroup_X, DataGroup_testYb = \
            Mymodel_data_extraction(target_combination, trainXn, trainY, trainYb, testXn, testY, testYb, Xcolname)
        # training
        Mymodel_f = Mymodel_training(prefix, target_combination, DataGroup_X, DataGroup_Yb)
        # prediction
        #Mymodel_f = prefix + "/control/model"+target_combination[0]+target_combination[1]+"_Full_1v1.h5"
        Mymodel = load_model(Mymodel_f)
        myauc, myap, myf1, mymcc, myacc, myPred_prob_alltest = Mymodel_Predict(Mymodel, testDataGroup_X,
                                                                               DataGroup_testYb, testXn)
        aucs[i] = myauc
        aps[i] = myap
        f1s[i] = myf1
        mccs[i] = mymcc
        accs[i] = myacc
        PPAs.append(myPred_prob_alltest)
        model_target = target_combination
        mymodels.append(model_target)
        print(model_target)

    print(time.time() - t0)

    df_aucs = pd.DataFrame(aucs)
    df_aps = pd.DataFrame(aps)
    df_f1s = pd.DataFrame(f1s)
    df_mccs = pd.DataFrame(mccs)
    df_accs = pd.DataFrame(accs)

    df_aucs.to_csv(prefix + "/control/Control_1v1_aucs_Upsampling_Test_20210920.csv")
    df_aps.to_csv(prefix + "/control/Control_1v1_aps_Upsampling_Test_20210920.csv")
    df_f1s.to_csv(prefix + "/control/Control_1v1_f1s_Upsampling_Test_20210920.csv")
    df_mccs.to_csv(prefix + "/control/Control_1v1_mccs_Upsampling_Test_20210920.csv")
    df_accs.to_csv(prefix + "/control/Control_1v1_accs_Upsampling_Test_20210920.csv")

    return df_aucs, df_aps, df_f1s, df_mccs, df_accs, mymodels, PPAs


# integration
def Cscorescalculating(prefix, mymodels, PPAs, testXn):
    Cancerlist = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    n = 7  # the types of cancers

    Cscores = []
    Cscore_models = []
    for CancerType in Cancerlist:
        Cscore, Cscore_model = Calculate_Cscores_1D(mymodels, PPAs, CancerType)
        Cscore = Cscore.iloc[:, 1:]
        Cscore_model = S_T_NameFormation_1D(Cscore_model)
        Cscore_models.append(Cscore_model)
        Cscore.columns = Cscore_model
        Cscores.append(Cscore)
    '''
    for i in range(len(Cscores)):
        Ctempt = Cscores[i]
        filename = prefix + "/control/Cscores" + str(i) + ".csv"
        Ctempt.to_csv(filename, header=True)
    '''
    ### mean
    Cscore_means = np.zeros((len(testXn), len(Cancerlist)))
    for i in range(len(Cancerlist)):
        Cscore_means[:, i] = Cscores[i].mean(1)

    Cscore_means_print = pd.DataFrame(Cscore_means,
                                      columns=["Cscores0_mean", "Cscores1_mean", "Cscores2_mean", "Cscores3_mean",
                                               "Cscores4_mean", "Cscores5_mean", "Cscores6_mean"])
    Cscore_means_print.to_csv(prefix + '/control/Cscore_integration.csv', header=True)
    return Cscore_means

def Prediction(prefix, Cscore_means, testXn, testY):
    #### predict with test dataset
    predY = [np.argmax(Cscore_means[i]) for i in range(len(testXn))]
    predY = np.array(predY)
    print("The acc of integration is", accuracy_score(testY, predY))

    t = pd.DataFrame(testY, columns=['testY'])
    p = pd.DataFrame(predY, columns=['predY'])

    results = t.join(p)
    results.to_csv(prefix + '/control/TL_1v1_Upsampling_test_pred_20210926.csv', header=True)
    return predY

def CscoreScaling(prefix, Cscore_means, testXn, testY, predY):
    # scaled to 1
    Cscore_means_scaled = np.zeros((Cscore_means.shape))
    Sample_sum_score = np.sum(Cscore_means, axis=1)
    for i in range(len(testXn)):
        Cscore_means_scaled[i] = Cscore_means[i] / Sample_sum_score[i]
    Cscore_means_scaled_print = pd.DataFrame(Cscore_means_scaled,
                                             columns=["C0n", "C1n", "C2n", "C3n", "C4n", "C5n", "C6n"])
    Cscore_means_scaled_print["Max"] = np.max(Cscore_means_scaled, axis=1)
    Cscore_means_scaled_print["Label"] = testY
    Cscore_means_scaled_print["Predict"] = predY

    Cscore_means_scaled_print.to_csv(prefix + '/control/Cscore_integration_scaled2.csv', header=True)
    return Cscore_means_scaled


def CancerDL(prefix, trainXn_UP, trainY_UP, trainYb_UP, testXn, testY, testYb, Xcolname):
    mkdir(prefix)
    #sys.stdout = Logger(prefix + '/control/CancerDL.log', sys.stdout)
    #sys.stderr = Logger(prefix + '/control/CancerDL.log_file', sys.stderr)
    df_aucs, df_aps, df_f1s, df_mccs, df_accs, mymodels, PPAs = one_fold(prefix, trainXn_UP, trainY_UP, trainYb_UP,
                                                                         testXn, testY, testYb, Xcolname)
    Cscore_means = Cscorescalculating(prefix, mymodels, PPAs, testXn)
    predY = Prediction(prefix, Cscore_means, testXn, testY)
    Cscore_means_scaled = CscoreScaling(prefix, Cscore_means, testXn, testY, predY)
    Integrated_auc, Integrated_ap, Integrated_f1, Integrated_mcc, Integrated_acc = Auc_Ap_F1_Mcc_Acc_Evaluation(Cscore_means_scaled, predY, testY, testYb)
    print("auc: ", Integrated_auc)
    print("ap: ", Integrated_ap)
    print("f1: ", Integrated_f1)
    print("mcc: ", Integrated_mcc)
    print("acc: ", Integrated_acc)
