#Input:
#OutputFile:
#Plot:
#For Shapley value: error because we have 41 features that is more than the maximal_feature=16.
#For Owen value:

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
from keras.models import load_model
import sys
import pickle

prefix = '../../03results/version6/'

CIndex = sys.argv[1]
CIndex = int(CIndex)

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
CompareType = "1v1" ## extendatble

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
    Group = Permutation[CIndex]  ## change1 LICR 13/LIUG 17
    DataGroup2_X, DataGroup2_Yb = Extract_DataGroup_1v1(Xn,Y,Yb,Xcolname,Group)

    print(Group)
    #CancerType = 'BRCR' ##change2

    Xn = DataGroup2_X
    Y = DataGroup2_Yb


#mycombination = "C2C0_C2C6" # e.g.C0C1_C0C1 #compare C0C1 #change3
mycombination = Permutation[CIndex][0]+Permutation[CIndex][1] #


#%%
import shap

T1 = np.zeros((100, 41))
T2 = np.zeros(100)
shap_values = shap.Explanation(T1,T2,T1)
#%%
for i in range(10):
    fold = 'fold'+ str(i)
    # prediction
    #model_f = prefix+ fold + '/models/Final_model'+ mycombination +"_Full_1v1.h5"
    model_f = prefix + fold + '/control/model' + mycombination + "_Full_1v1.h5"
    model = load_model(model_f)

    #--------------Tabular data with independent (Shapley value) masking---------------
    # build a Permutation explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Permutation(model.predict_proba, Xn)
    if (len(Xn) > 100):
        shap_values_foldi = explainer(Xn[:100])

        # get just the explanations for the positive class
        shap_values_foldi = shap_values_foldi[..., 1]

        shap_values = shap_values + shap_values_foldi
        print("Finished fold", i)
    else:
        LXn = len(Xn)
        shap_values_foldi = explainer(Xn[:LXn])

        # get just the explanations for the positive class
        shap_values_foldi = shap_values_foldi[..., 1]

        shap_values = shap_values + shap_values_foldi
        print("Finished fold", i)

#%%
shap_values = shap_values/10 ###
shap_values.feature_names = Myfeatures
#%% save
# dump
filename = './ShapValues_' + mycombination + '.pk'
with open(filename, 'wb') as f:
  pickle.dump(shap_values, f)
#%% load
#with open(filename, 'rb') as f:
#  shap_values = pickle.load(f)
#%% Plot a global summary
shap.plots.bar(shap_values, show = False)
plt.savefig('./shap_bar_Shapley_10fold_Average_control_'+ mycombination + '.pdf') #change4

'''
#%% Plot a single instance
shap.plots.waterfall(shap_values[0], show = False)
plt.savefig('./shap_waterfall_Shapley.png')
'''
'''
#----------------------------owen-------------------------
#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
print(le.classes_)
y = le.transform(y)
#%%
# build a clustering of the features based on shared information about y
clustering = shap.utils.hclust(X, y)

# above we implicitly used shap.maskers.Independent by passing a raw dataframe as the masker
# now we explicitly use a Partition masker that uses the clustering we just computed
masker = shap.maskers.Partition(X, clustering=clustering)

# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, masker)
#%%
shap_values2 = explainer(X[:100])

# get just the explanations for the positive class
shap_values2 = shap_values2[...,1]


shap.plots.bar(shap_values2, show = False)
plt.savefig('./shap_bar_Owen.png')
shap.plots.waterfall(shap_values2[0], show = False)
plt.savefig('./shap_waterfall_Owen.png')
'''



