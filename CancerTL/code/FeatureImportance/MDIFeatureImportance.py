#Input:
#OutputFile:
#Plot:

print(__doc__)
import matplotlib.pyplot as plt

# Data generation and model fitting
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn import model_selection

prefix = '../../03results/version6/'

def preprocess(feature, label):
    # labelencoder
    le = LabelEncoder()
    le.fit(label)
    print(list(le.classes_))
    label = le.transform(label)
    label = label.astype(np.int32)
    feature = np.array(feature)
    # onehot encoder
    y_onehot = to_categorical(label)
    X = feature
    Y = label
    Yb = y_onehot

    # standarlization
    Scaler = StandardScaler()
    Scaler.fit(X)
    Xn = Scaler.transform(X)

    return Xn, Y, Yb


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
    DataGroup_X = DataGroup.iloc[:, :41]  # features
    DataGroup_Yb = DataGroup.loc[:, combination]  # onehot label
    return DataGroup_X, DataGroup_Yb

Cancerlist = ["C0","C1","C2","C3","C4","C5","C6"]
n = 7 # the types of cancers
Permutation = CancerTypeCombination_generator("C0","C1","C2","C3","C4","C5","C6",k=2)

df = pd.read_csv("../../01data/mydata.csv")
#%%
Myfeatures = ('OmegaScore', 'AFP', 'Angiopoietin-2', 'AXL',
       'CA-125', 'CA 15-3', 'CA19-9', 'CD44',
       'CEA', 'CYFRA 21-1', 'DKK1', 'Endoglin',
       'FGF2', 'Follistatin', 'Galectin-3',
       'G-CSF', 'GDF15', 'HE4', 'HGF',
       'IL-6', 'IL-8', 'Kallikrein-6',
       'Leptin', 'Mesothelin', 'Midkine',
       'Myeloperoxidase', 'NSE', 'OPG', 'OPN',
       'PAR', 'Prolactin', 'sEGFR', 'sFas',
       'SHBG', 'sHER2/sEGFR2/sErbB2', 'sPECAM-1',
       'TGFa', 'Thrombospondin-2', 'TIMP-1',
       'TIMP-2', 'Gender')

#%% data preprocess
CompareType = "1v1" ##change

if CompareType == "1v1":
    # 1v1
    # For binary classification.(2 cancer type)
    #X = df.drop(['Sex', 'Tumor type'], axis=1)  ## gender/Sex
    X = df.drop(['Tumor type'], axis=1) ## gender/Sex
    y = df['Tumor type']
    y = np.array(y)
    Xcolname = X.columns.values.tolist()

    Xn, Y, Yb = preprocess(X,y) #trainYb does not be used in the further anaysis

    #Permutation = deduplicate(Permutation) ########dedup
    L= len(Permutation)
    Group = Permutation[17] ## change LICR 13/LIUG 17
    DataGroup2_X, DataGroup2_Yb = Extract_DataGroup_1v1(Xn,Y,Yb,Xcolname,Group)

    print(Group)
    CancerType = 'LIUG' ##change

    Xn = DataGroup2_X
    Y = DataGroup2_Yb
elif CompareType == "1vR":
    # For 1vR classification.(7 cancer types)
    # ['Breast', 'Colorectum', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Upper GI']

    CancerType = 'Breast' ## change
    df2 = df.copy()
    df2["CancerType"] = np.where(df2["Tumor type"] == CancerType,1,0)
    #X = df2.drop(['Sex', 'Tumor type', 'AJCC Stage', "CancerType"], axis=1)  ## gender/Sex
    X = df2.drop(['Tumor type', "CancerType"], axis=1) ## gender/Sex
    y = df2["CancerType"]
    y = np.array(y)
    Xcolname = X.columns.values.tolist()

    Xn, Y, Yb = preprocess(X,y) #trainYb does not be used in the further anaysis

#%%
from sklearn.ensemble import RandomForestClassifier

feature_names = [f'feature {i}' for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(Xn, Y)

#——————————————————————————————————————————————————————————————————————————————
# Feature importance based on mean decrease in impurity
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
#forest_importances.plot.bar(ax=ax)
plt.bar(range(41),importances)
ax.set_title("Feature importances of classification of " + CancerType + " cancer")
ax.set_ylabel("Mean decrease in impurity")
ticks = np.arange(0,41,1)
plt.xticks(ticks, Myfeatures, rotation=90, size=8)
plt.ylim(0,0.16)
plt.savefig(prefix + 'viz/FI_' + CancerType +'.pdf')
plt.show()




