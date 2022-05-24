import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from scipy import stats
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

def CancerAB(i, trainXn_UP, trainY_UP, testXn, testY):
    # --------------------2. Model ------------------------------
    GB = AdaBoostClassifier(learning_rate=1.0, random_state = 0)
    GB.fit(trainXn_UP, trainY_UP)

    # %%
    # predict
    GB_predY = GB.predict(testXn)
    GB_acc = metrics.accuracy_score(testY, GB_predY)
    print("test accuracy =", GB_acc)
    GB_predY_prob = GB.predict_proba(testXn)
    # %%
    # one hot
    le = LabelEncoder()
    le.fit(testY)
    list(le.classes_)
    testY = le.transform(testY)
    testY = testY.astype(np.int32)

    onehot_testY = to_categorical(testY)

    # %%
    Integrated_auc = roc_auc_score(onehot_testY, GB_predY_prob, multi_class="ovo", average="micro")

    Integrated_ap = average_precision_score(onehot_testY, GB_predY_prob, average="micro")

    Integrated_f1 = f1_score(testY, GB_predY, average='micro')  ###
    Integrated_mcc = matthews_corrcoef(testY, GB_predY)
    Integrated_acc = accuracy_score(testY, GB_predY)

    print("auc: ", Integrated_auc)
    print("ap: ", Integrated_ap)
    print("f1: ", Integrated_f1)
    print("mcc: ", Integrated_mcc)
    print("acc: ", Integrated_acc)
