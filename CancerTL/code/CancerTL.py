# 1 vs 1
# 19 Aug, 2021
# split the data into train, valid, test.(8:1:1)
# the test_group transfer into valid_group

from Models_utilities import *
import numpy as np
import pandas as pd
import time
from keras import backend as K
import os


# %%
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.mkdir(path)
        print("--- new folder... ---")
        print("--- OK ---")

    else:
        print("--- There is this folder! ---")

    subfolder = os.path.exists(path + '/models')
    if not subfolder:
        os.mkdir(path + '/models')
        print("--- new subfolder... ---")
        print("--- OK ---")
    else:
        print("--- There is this subfolder! ---")


# %%

def Mymodel_data_extraction(source, target, trainXn_UP, trainY_UP, trainYb_UP, validXn, validY, validYb, Xcolname):
    DataGroup1_X, DataGroup1_Yb = Extract_DataGroup_1v1(trainXn_UP, trainY_UP, trainYb_UP, Xcolname, source)
    DataGroup2_X, DataGroup2_Yb = Extract_DataGroup_1v1(trainXn_UP, trainY_UP, trainYb_UP, Xcolname, target)
    validDataGroup2_X, DataGroup2_validYb = Extract_DataGroup_1v1(validXn, validY, validYb, Xcolname, target)
    return DataGroup1_X, DataGroup1_Yb, DataGroup2_X, DataGroup2_Yb, validDataGroup2_X, DataGroup2_validYb


# Transfer learning
def Mymodel_training(prefix, source, target, DataGroup1_X, DataGroup1_Yb, DataGroup2_X, DataGroup2_Yb):
    K.clear_session()
    pretrain_model_save_filename = prefix + '/models/Pretrain_model' + source[0] + source[1] + "_" + target[0] + target[1] + "_Full_1v1.h5"

    # model_0 : 2+3 (v2)
    Pretrain_model(DataGroup1_X, DataGroup1_Yb, pretrain_model_save_filename)
    model = Freeze_Layer(pretrain_model_save_filename)
    model2 = Finetune_model(DataGroup2_X, DataGroup2_Yb, model)

    Final_model_filename = prefix + '/models/Final_model' + source[0] + source[1] + "_" + target[0] + target[1] + "_Full_1v1.h5"
    model2.save(Final_model_filename)
    return Final_model_filename


# Select Selections and predict
def Mymodel_predict(model2, validDataGroup2_X, DataGroup2_validYb, testXn):
    DataGroup2_validY = [np.argmax(one_hot) for one_hot in np.array(DataGroup2_validYb)]
    auc, ap, f1, mcc, acc = Auc_Ap_F1_Mcc_Acc(model2, validDataGroup2_X, DataGroup2_validY, DataGroup2_validYb)
    Pred_prob_test = Pred_all_cancer_test(model2, testXn)  # testXn including all cancer types
    return auc, ap, f1, mcc, acc, Pred_prob_test

    # ------------------------------------main----------------------------------


Permutation = CancerTypeCombination_generator("C0", "C1", "C2", "C3", "C4", "C5", "C6", k=2)
L = len(Permutation)


def one_fold(prefix, trainXn_UP, trainY_UP, trainYb_UP, validXn, validY, validYb, testXn, Xcolname):
    aucs = np.zeros((L, L))
    aps = np.zeros((L, L))
    f1s = np.zeros((L, L))
    mccs = np.zeros((L, L))
    accs = np.zeros((L, L))
    PPAs = []
    models = []

    t0 = time.time()
    for i in range(L):
        for j in range(L):
            source_combination = Permutation[i]  ######
            target_combination = Permutation[j]  ######
            #
            # myauc, myacc, Pred_prob_test = Mymodel(source_combination,target_combination)
            # data extraction

            DataGroup1_X, DataGroup1_Yb, DataGroup2_X, DataGroup2_Yb, validDataGroup2_X, DataGroup2_validYb = \
                Mymodel_data_extraction(source_combination, target_combination, trainXn_UP, trainY_UP, trainYb_UP,
                                        validXn, validY, validYb, Xcolname)
            # training

            #Mymodel_f = Mymodel_training(prefix, source_combination, target_combination, DataGroup1_X, DataGroup1_Yb,DataGroup2_X, DataGroup2_Yb)


            # prediction
            Mymodel_f = prefix+'/models/Final_model'+source_combination[0]+source_combination[1]+"_"+target_combination[0]+target_combination[1]+"_Full_1v1.h5"
            Mymodel = load_model(Mymodel_f)
            myauc, myap, myf1, mymcc, myacc, Pred_prob_test = Mymodel_predict(Mymodel, validDataGroup2_X,
                                                                              DataGroup2_validYb,
                                                                              testXn)  # valid dataset and test dataset

            aucs[i][j] = myauc
            aps[i][j] = myap
            f1s[i][j] = myf1
            mccs[i][j] = mymcc
            accs[i][j] = myacc
            PPAs.append(Pred_prob_test)
            model = (source_combination, target_combination)
            models.append(model)
            print(model)
    print(time.time() - t0)
    models = np.array(models)
    # save file
    df_aucs = pd.DataFrame(aucs, columns=Permutation, index=Permutation)
    df_aps = pd.DataFrame(aps, columns=Permutation, index=Permutation)
    df_f1s = pd.DataFrame(f1s, columns=Permutation, index=Permutation)
    df_mccs = pd.DataFrame(mccs, columns=Permutation, index=Permutation)
    df_accs = pd.DataFrame(accs, columns=Permutation, index=Permutation)

    df_aucs.to_csv(prefix + '/TL_1v1_aucs_Test.csv', header=True)
    df_aps.to_csv(prefix + '/TL_1v1_aps_Test.csv', header=True)
    df_f1s.to_csv(prefix + '/TL_1v1_f1s_Test.csv', header=True)
    df_mccs.to_csv(prefix + '/TL_1v1_mccs_Test.csv', header=True)
    df_accs.to_csv(prefix + '/TL_1v1_accs_Test.csv', header=True)

    return df_aucs, df_aps, df_f1s, df_mccs, df_accs, models, PPAs


def aucs_max(df_aucs, PPAs):
    models_max = []
    PPAs_max = []

    # select best source(model) for each target binary classification model (according auc)
    for i in Permutation:
        imax = df_aucs.loc[:][i].idxmax()  ###select best models according to auc
        model = (imax, i)
        models_max.append(model)
        imax_index = Permutation.index(imax)
        i_index = Permutation.index(i)
        PPA_index = imax_index * L + i_index  ## PPA were predict by test dataset
        PPAs_max.append(PPAs[PPA_index])
    return models_max, PPAs_max


# predict + integration using test dataset
def Cscorescalculating(prefix, models_max, PPAs_max, testXn):
    Cancerlist = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    n = 7  # the types of cancers

    Cscores = []
    Cscore_models = []

    # calculate Cscore for each type of cancer
    for CancerType in Cancerlist:
        # Cscore, Cscore_model = Calculate_Cscores(models, PPAs, CancerType) ALL 42*42 (source*target)
        Cscore, Cscore_model = Calculate_Cscores(models_max, PPAs_max, CancerType)  # max 1*42
        Cscore = Cscore.iloc[:, 1:]
        Cscore_model = S_T_NameFormation(Cscore_model)
        Cscore_models.append(Cscore_model)
        Cscore.columns = Cscore_model
        Cscores.append(Cscore)
    # print detail
    for i in range(len(Cscores)):
        Ctempt = Cscores[i]
        filename = prefix+'/Cscores_max'+str(i)+".csv"
        Ctempt.to_csv(filename,header=True)

    ### mean
    Cscore_means = np.zeros((len(testXn), len(Cancerlist)))
    for i in range(len(Cancerlist)):
        Cscore_means[:, i] = Cscores[i].mean(1)

    Cscore_means_print = pd.DataFrame(Cscore_means,
                                      columns=["Cscores0_mean", "Cscores1_mean", "Cscores2_mean", "Cscores3_mean",
                                               "Cscores4_mean", "Cscores5_mean", "Cscores6_mean"])
    Cscore_means_print.to_csv(prefix + '/Cscore_integration.csv', header=True)
    return Cscore_means


def Prediction(prefix, Cscore_means, testXn, testY):
    #### predict with test dataset
    predY = [np.argmax(Cscore_means[i]) for i in range(len(testXn))]
    predY = np.array(predY)
    print("The acc of integration is", accuracy_score(testY, predY))

    t = pd.DataFrame(testY, columns=['testY'])
    p = pd.DataFrame(predY, columns=['predY'])

    results = t.join(p)
    results.to_csv(prefix + '/TL_1v1_Upsampling_test_pred.csv', header=True)
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

    Cscore_means_scaled_print.to_csv(prefix + '/Cscore_integration_scaled2.csv', header=True)
    return Cscore_means_scaled


def CancerTL(prefix, trainXn_UP, trainY_UP, trainYb_UP, validXn, validY, validYb, testXn, testY, testYb, Xcolname):
    mkdir(prefix)
    df_aucs, df_aps, df_f1s, df_mccs, df_accs, models, PPAs = one_fold(prefix, trainXn_UP, trainY_UP, trainYb_UP, validXn, validY, validYb, testXn, Xcolname)
    # predict + integration using test dataset
    models_max, PPAs_max = aucs_max(df_aucs, PPAs)
    Cscore_means = Cscorescalculating(prefix, models_max, PPAs_max, testXn)
    predY = Prediction(prefix, Cscore_means, testXn, testY)
    Cscore_means_scaled = CscoreScaling(prefix, Cscore_means, testXn, testY, predY)

    Integrated_auc, Integrated_ap, Integrated_f1, Integrated_mcc, Integrated_acc = Auc_Ap_F1_Mcc_Acc_Evaluation(Cscore_means_scaled, predY, testY, testYb)
    print("auc: ", Integrated_auc)
    print("ap: ", Integrated_ap)
    print("f1: ", Integrated_f1)
    print("mcc: ", Integrated_mcc)
    print("acc: ", Integrated_acc)

