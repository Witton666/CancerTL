# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Visualization_utilities as Vu
import seaborn as sns

prefix = '../results/version1/'

mylabel = ['(BR, CR)', '(BR, LI)', '(BR, LU)', '(BR, OV)', '(BR, PA)',
           '(BR, UG)', '(CR, BR)', '(CR, LI)', '(CR, LU)', '(CR, OV)',
           '(CR, PA)', '(CR, UG)', '(LI, BR)', '(LI, CR)', '(LI, LU)',
           '(LI, OV)', '(LI, PA)', '(LI, UG)', '(LU, BR)', '(LU, CR)',
           '(LU, LI)', '(LU, OV)', '(LU, PA)', '(LU, UG)', '(OV, BR)',
           '(OV, CR)', '(OV, LI)', '(OV, LU)', '(OV, PA)', '(OV, UG)',
           '(PA, BR)', '(PA, CR)', '(PA, LI)', '(PA, LU)', '(PA, OV)',
           '(PA, UG)', '(UG, BR)', '(UG, CR)', '(UG, LI)', '(UG, LU)',
           '(UG, OV)', '(UG, PA)']

mysource = ['[BR, CR]', '[BR, LI]', '[BR, LU]', '[BR, OV]', '[BR, PA]',
            '[BR, UG]', '[CR, BR]', '[CR, LI]', '[CR, LU]', '[CR, OV]',
            '[CR, PA]', '[CR, UG]', '[LI, BR]', '[LI, CR]', '[LI, LU]',
            '[LI, OV]', '[LI, PA]', '[LI, UG]', '[LU, BR]', '[LU, CR]',
            '[LU, LI]', '[LU, OV]', '[LU, PA]', '[LU, UG]', '[OV, BR]',
            '[OV, CR]', '[OV, LI]', '[OV, LU]', '[OV, PA]', '[OV, UG]',
            '[PA, BR]', '[PA, CR]', '[PA, LI]', '[PA, LU]', '[PA, OV]',
            '[PA, UG]', '[UG, BR]', '[UG, CR]', '[UG, LI]', '[UG, LU]',
            '[UG, OV]', '[UG, PA]']

L = len(mylabel)


# %%
def Input(Evaluation_fileName):
    FileName = prefix + Evaluation_fileName
    Evaluation = pd.read_csv(FileName, index_col=0, header=0)
    return Evaluation


#
def EvaluationMax_10Fold(EvaluationType, group):
    Evaluation_10Fold = pd.DataFrame(np.zeros((10, 2 + L)))
    tempt = ['Fold', 'Statistics']
    tempt.extend(mylabel)
    Evaluation_10Fold.columns = tempt
    Evaluation_10Fold['Fold'] = ['fold' + str(i) for i in range(10)]
    if (group == 'CancerTL'):
        Evaluation_10Fold['Statistics'] = ['CancerTL' for i in range(10)]
        for i in range(10):
            FileName = 'fold' + str(i) + '/TL_1v1_' + EvaluationType + '_Test.csv'
            Evaluation = Input(FileName)
            # Evaluation_10Fold.iloc[0, 2:] = Evaluation.mean(axis=0)
            Evaluation_10Fold.iloc[i, 2:] = Evaluation.max()
    elif (group == 'CancerTL-mean'):
        Evaluation_10Fold['Statistics'] = ['CancerTL-mean' for i in range(10)]
        for i in range(10):
            FileName = 'fold' + str(i) + '/TL_1v1_' + EvaluationType + '_Test.csv'
            Evaluation = Input(FileName)
            Evaluation_10Fold.iloc[i, 2:] = Evaluation.mean(axis=0)
            # Evaluation_10Fold.iloc[i, 2:] = Evaluation.max()
    elif (group == 'CancerDL'):
        Evaluation_10Fold['Statistics'] = ['CancerDL' for i in range(10)]
        for i in range(10):
            FileName = 'fold' + str(i) + '/control/Control_1v1_' + EvaluationType + '_Upsampling_Test.csv'
            Evaluation = pd.read_csv(prefix + FileName, index_col=0, header=0)
            Evaluation_10Fold.iloc[i, 2:] = Evaluation.iloc[:, 0]

    return Evaluation_10Fold


# equal to flatten function
def EvaluationResults_reshape(EvaluationResults, mylabel, n_one_round):
    L = len(EvaluationResults)
    n = len(mylabel)
    # EvaluationRaw = np.array(EvaluationResults.iloc[:, 2:-1]) #including mean row
    EvaluationRaw = np.array(EvaluationResults.iloc[:, 2:])
    AccValues = EvaluationRaw.reshape(n_one_round * n, order='F')
    temp = EvaluationResults.iloc[:, 0:2]
    ReshapedResults = pd.DataFrame(np.zeros((n * L, 4)))
    for i in range(n):
        ReshapedResults.iloc[n_one_round * i:n_one_round * (i + 1), 0:2] = temp
        ReshapedResults.iloc[n_one_round * i:n_one_round * (i + 1), 2] = [mylabel[i] for j in range(n_one_round)]
    ReshapedResults.iloc[:, 3] = AccValues
    ReshapedResults.columns = ["Fold", "Statistics", "X", "Value"]
    return ReshapedResults


# bar plot
def ComparisonsBinary(Dataset, EvaluationType):
    fig, ax = plt.subplots(figsize=(14, 7))
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(font_scale=1.5, style="ticks", rc=custom_params)

    g = sns.barplot(x="X", y="Value", hue='Statistics', data=Dataset, errwidth=0.5, capsize=0.05)
    g.legend_.set_title(None)

    ax.set_title('The comparisons of ' + EvaluationType + ' for CancerTL and CancerDL', size=20)
    ax.set_xlabel('')  ####
    ax.set_ylabel(EvaluationType, size=16)  ####
    plt.xticks(rotation=90, size=14)
    plt.yticks(size=15)
    plt.subplots_adjust(bottom=0.14)
    plt.ylim(0.75, 1.0)
    plt.legend(loc="lower left")
    plt.savefig(prefix + 'viz/Fig4B_' + EvaluationType + '.pdf')
    plt.show()


# heatmap binary; i is fold index
def heatmap1fold(i, EvaluationType):  # lower case
    FileName = 'fold' + str(i) + '/TL_1v1_' + EvaluationType + '.csv'
    Evaluation = Input(FileName)
    mat = np.zeros((L, L))
    Evaluation2 = np.array(Evaluation)
    ymax_index = np.argmax(Evaluation2, axis=0)  # each columns
    for q in range(L):
        a = ymax_index[q]
        ymax_value = Evaluation2[a, q]
        mat[a, q] = ymax_value
    mat = pd.DataFrame(mat, columns=mylabel, index=mysource)
    pd.DataFrame(mat).to_csv(prefix + 'viz/fold' + str(i) + 'TL_1v1_' + EvaluationType + '_Test_max.csv')

    # all max
    act = (Evaluation == np.max(Evaluation))
    mat_all = act * Evaluation
    pd.DataFrame(mat_all).to_csv(
        prefix + 'viz/fold' + str(i) + 'TL_1v1_' + EvaluationType + '_Test_allmax.csv')

    # plot
    sns.set(font_scale=1)
    sns.clustermap(Evaluation, row_cluster=False, col_cluster=False, xticklabels=True, yticklabels=True, cmap='Blues')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '1.pdf')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '1.png')
    plt.show()

    sns.set(font_scale=1)
    sns.clustermap(mat_all, row_cluster=False, col_cluster=False, xticklabels=True, yticklabels=True, cmap='Blues')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '2.pdf')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '2.png')
    plt.show()

    sns.set(font_scale=1)
    sns.clustermap(mat, row_cluster=False, col_cluster=False, xticklabels=True, yticklabels=True, cmap='Blues')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '3.pdf')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapFig' + EvaluationType + '3.png')
    plt.show()
#%%
def heatmap1foldTLminusDL(i, EvaluationType):
    FileName = 'fold' + str(i) + '/TL_1v1_' + EvaluationType + '_Test_20210920.csv'
    Evaluation = Input(FileName)
    controlfile = prefix + 'fold' + str(i) + '/control/Control_1v1_' + EvaluationType + '_Upsampling_Test.csv'
    contr_df = pd.read_csv(controlfile, index_col=0, header=0)
    diff = np.array(Evaluation)-np.array(contr_df.T)
    diff = pd.DataFrame(diff,index=mysource, columns=mylabel)
    sns.set(font_scale=1)
    cmap = sns.diverging_palette(250, 15, as_cmap=True)
    sns.clustermap(diff, row_cluster=False, col_cluster=False, xticklabels=True, yticklabels=True, cmap=cmap, center=-0.01)
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapDiffFig' + EvaluationType + '1.pdf')
    plt.savefig(prefix + 'viz/fold' + str(i) + 'HeatmapDiffFig' + EvaluationType + '1.png')
    plt.show()
# %%
# CancerTL
DataAccsmax = EvaluationMax_10Fold('accs', 'CancerTL')
DataAucsmax = EvaluationMax_10Fold('aucs', 'CancerTL')
DataApsmax = EvaluationMax_10Fold('aps', 'CancerTL')
DataF1smax = EvaluationMax_10Fold('f1s', 'CancerTL')
DataMccsmax = EvaluationMax_10Fold('mccs', 'CancerTL')

DataAccsmaxReshape = EvaluationResults_reshape(DataAccsmax, mylabel, 10)
DataAucsmaxReshape = EvaluationResults_reshape(DataAucsmax, mylabel, 10)
DataApsmaxReshape = EvaluationResults_reshape(DataApsmax, mylabel, 10)
DataF1smaxReshape = EvaluationResults_reshape(DataF1smax, mylabel, 10)
DataMccsmaxReshape = EvaluationResults_reshape(DataMccsmax, mylabel, 10)

# CancerTL-mean
DataAccsmean = EvaluationMax_10Fold('accs', 'CancerTL-mean')
DataAucsmean = EvaluationMax_10Fold('aucs', 'CancerTL-mean')
DataApsmean = EvaluationMax_10Fold('aps', 'CancerTL-mean')
DataF1smean = EvaluationMax_10Fold('f1s', 'CancerTL-mean')
DataMccsmean = EvaluationMax_10Fold('mccs', 'CancerTL-mean')

## flatten
DataAccsmeanReshape = EvaluationResults_reshape(DataAccsmean, mylabel, 10)
DataAucsmeanReshape = EvaluationResults_reshape(DataAucsmean, mylabel, 10)
DataApsmeanReshape = EvaluationResults_reshape(DataApsmean, mylabel, 10)
DataF1smeanReshape = EvaluationResults_reshape(DataF1smean, mylabel, 10)
DataMccsmeanReshape = EvaluationResults_reshape(DataMccsmean, mylabel, 10)
# %%
# Control
ControlAccs = EvaluationMax_10Fold('accs', 'CancerDL')
ControlAucs = EvaluationMax_10Fold('aucs', 'CancerDL')
ControlAps = EvaluationMax_10Fold('aps', 'CancerDL')
ControlF1s = EvaluationMax_10Fold('f1s', 'CancerDL')
ControlMccs = EvaluationMax_10Fold('mccs', 'CancerDL')

ControlAccsReshape = EvaluationResults_reshape(ControlAccs, mylabel, 10)
ControlAucsReshape = EvaluationResults_reshape(ControlAucs, mylabel, 10)
ControlApsReshape = EvaluationResults_reshape(ControlAps, mylabel, 10)
ControlF1sReshape = EvaluationResults_reshape(ControlF1s, mylabel, 10)
ControlMccsReshape = EvaluationResults_reshape(ControlMccs, mylabel, 10)

'''
#control3l
Control3lAccs = EvaluationMax_10Fold('accs', 'CancerDL3l')
Control3lAucs = EvaluationMax_10Fold('aucs', 'CancerDL3l')
Control3lAps = EvaluationMax_10Fold('aps', 'CancerDL3l')
Control3lF1s = EvaluationMax_10Fold('f1s', 'CancerDL3l')
Control3lMccs = EvaluationMax_10Fold('mccs', 'CancerDL3l')

Control3lAccsReshape = EvaluationResults_reshape(Control3lAccs, mylabel, 10)
Control3lAucsReshape = EvaluationResults_reshape(Control3lAucs, mylabel, 10)
Control3lApsReshape = EvaluationResults_reshape(Control3lAps, mylabel, 10)
Control3lF1sReshape = EvaluationResults_reshape(Control3lF1s, mylabel, 10)
Control3lMccsReshape = EvaluationResults_reshape(Control3lMccs, mylabel, 10)
'''
# %%
# Summary
DataAccsmaxcontrol = pd.concat([ControlAccsReshape, DataAccsmaxReshape], axis=0)
DataAucsmaxcontrol = pd.concat([ControlAucsReshape, DataAucsmaxReshape], axis=0)
DataApsmaxcontrol = pd.concat([ControlApsReshape, DataApsmaxReshape], axis=0)
DataF1smaxcontrol = pd.concat([ControlF1sReshape, DataF1smaxReshape], axis=0)
DataMccsmaxcontrol = pd.concat([ControlMccsReshape, DataMccsmaxReshape], axis=0)

DataAccsmaxcontrol.to_csv("DataAccsmaxcontrol.csv")
DataAucsmaxcontrol.to_csv("DataAucsmaxcontrol.csv")
DataApsmaxcontrol.to_csv("DataApsmaxcontrol.csv")
DataF1smaxcontrol.to_csv("DataF1smaxcontrol.csv")
DataMccsmaxcontrol.to_csv("DataMccsmaxcontrol.csv")
#%%
# plot
ComparisonsBinary(DataAccsmaxcontrol, 'ACC')
ComparisonsBinary(DataAucsmaxcontrol, 'AUC')
ComparisonsBinary(DataApsmaxcontrol, 'AP')
ComparisonsBinary(DataF1smaxcontrol, 'F1')
ComparisonsBinary(DataMccsmaxcontrol, 'MCC')

# %%
DataSummax = DataAccsmax
DataSummax.iloc[:, 2:] = (DataAccsmax.iloc[:, 2:] + DataAucsmax.iloc[:, 2:] + DataApsmax.iloc[:, 2:] + DataF1smax.iloc[
                                                                                                       :,
                                                                                                       2:] + DataMccsmax.iloc[
                                                                                                             :, 2:]) / 5

# %%
# Evaluation heat map
for i in range(10):
    heatmap1fold(i, 'accs')
    heatmap1fold(i, 'aucs')
    heatmap1fold(i, 'aps')
    heatmap1fold(i, 'f1s')
    heatmap1fold(i, 'mccs')
# %%
for i in range(10):
    #heatmap1foldTLminusDL(i, 'accs')
    heatmap1foldTLminusDL(i, 'aucs')
    #heatmap1foldTLminusDL(i, 'aps')
    #heatmap1foldTLminusDL(i, 'f1s')
    #heatmap1foldTLminusDL(i, 'mccs')

# %%
# CancerTL
CancerTLBinary_EvaluationAcc = pd.DataFrame(
    {'Model': ['CancerTL' for k in range(10)], 'Type': ['ACC' for k in range(10)],
     'Value': DataAccsmax.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLBinary_EvaluationAuc = pd.DataFrame(
    {'Model': ['CancerTL' for k in range(10)], 'Type': ['AUC' for k in range(10)],
     'Value': DataAucsmax.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLBinary_EvaluationAp = pd.DataFrame(
    {'Model': ['CancerTL' for k in range(10)], 'Type': ['AUPRC' for k in range(10)],
     'Value': DataApsmax.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLBinary_EvaluationF1 = pd.DataFrame({'Model': ['CancerTL' for k in range(10)], 'Type': ['F1' for k in range(10)],
                                            'Value': DataF1smax.iloc[:, 2:].mean(axis=1),
                                            'Fold': [k for k in range(10)]})
CancerTLBinary_EvaluationMcc = pd.DataFrame(
    {'Model': ['CancerTL' for k in range(10)], 'Type': ['MCC' for k in range(10)],
     'Value': DataMccsmax.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
# CancerTL-mean
CancerTLMBinary_EvaluationAcc = pd.DataFrame(
    {'Model': ['CancerTL-mean' for k in range(10)], 'Type': ['ACC' for k in range(10)],
     'Value': DataAccsmean.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLMBinary_EvaluationAuc = pd.DataFrame(
    {'Model': ['CancerTL-mean' for k in range(10)], 'Type': ['AUC' for k in range(10)],
     'Value': DataAucsmean.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLMBinary_EvaluationAp = pd.DataFrame(
    {'Model': ['CancerTL-mean' for k in range(10)], 'Type': ['AUPRC' for k in range(10)],
     'Value': DataApsmean.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLMBinary_EvaluationF1 = pd.DataFrame(
    {'Model': ['CancerTL-mean' for k in range(10)], 'Type': ['F1' for k in range(10)],
     'Value': DataF1smean.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerTLMBinary_EvaluationMcc = pd.DataFrame(
    {'Model': ['CancerTL-mean' for k in range(10)], 'Type': ['MCC' for k in range(10)],
     'Value': DataMccsmean.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})

# CancerDL
CancerDLBinary_EvaluationAcc = pd.DataFrame(
    {'Model': ['CancerDL' for k in range(10)], 'Type': ['ACC' for k in range(10)],
     'Value': ControlAccs.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerDLBinary_EvaluationAuc = pd.DataFrame(
    {'Model': ['CancerDL' for k in range(10)], 'Type': ['AUC' for k in range(10)],
     'Value': ControlAucs.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerDLBinary_EvaluationAp = pd.DataFrame(
    {'Model': ['CancerDL' for k in range(10)], 'Type': ['AUPRC' for k in range(10)],
     'Value': ControlAps.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})
CancerDLBinary_EvaluationF1 = pd.DataFrame({'Model': ['CancerDL' for k in range(10)], 'Type': ['F1' for k in range(10)],
                                            'Value': ControlF1s.iloc[:, 2:].mean(axis=1),
                                            'Fold': [k for k in range(10)]})
CancerDLBinary_EvaluationMcc = pd.DataFrame(
    {'Model': ['CancerDL' for k in range(10)], 'Type': ['MCC' for k in range(10)],
     'Value': ControlMccs.iloc[:, 2:].mean(axis=1), 'Fold': [k for k in range(10)]})

Binary_Evaluation = CancerTLBinary_EvaluationAcc
mylist = [CancerTLBinary_EvaluationAuc, CancerTLBinary_EvaluationAp, CancerTLBinary_EvaluationF1,
          CancerTLBinary_EvaluationMcc,
          CancerTLMBinary_EvaluationAcc, CancerTLMBinary_EvaluationAuc, CancerTLMBinary_EvaluationAp,
          CancerTLMBinary_EvaluationF1,
          CancerTLMBinary_EvaluationMcc, CancerDLBinary_EvaluationAcc, CancerDLBinary_EvaluationAuc,
          CancerDLBinary_EvaluationAp,
          CancerDLBinary_EvaluationF1, CancerDLBinary_EvaluationMcc]
for a in mylist:
    Binary_Evaluation = Binary_Evaluation.append(a, ignore_index=True)

Binary_Evaluation.to_csv(prefix + 'viz/Binary_Evaluation.csv')  # df
# %% CancerTL model performance  barplot
# Draw a nested barplot by species and sex
sns.set(font_scale=1.5, rc={"xtick.bottom": True, "ytick.left": True})
sns.set_style("white")
g = sns.catplot(
    data=Binary_Evaluation, kind="bar",
    x="Type", y="Value", hue="Model",
    palette="colorblind", height=6, aspect=8/6, errwidth=1, capsize=0.05)
# g.despine()
g.set_axis_labels("", "Measurements")
plt.title("Performance comparisons of CancerTL and others")
plt.ylim(0, 1)
plt.savefig(prefix + 'viz/Fig4E.pdf')
plt.show()

# %% Integrated CancerTL performance auc
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# %% y_score
prefix = '../results/version6/'
# con = 'control/'
con = ''
Cscoresf0 = np.array(pd.read_csv(prefix + 'fold0/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf1 = np.array(pd.read_csv(prefix + 'fold1/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf2 = np.array(pd.read_csv(prefix + 'fold2/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf3 = np.array(pd.read_csv(prefix + 'fold3/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf4 = np.array(pd.read_csv(prefix + 'fold4/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf5 = np.array(pd.read_csv(prefix + 'fold5/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf6 = np.array(pd.read_csv(prefix + 'fold6/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf7 = np.array(pd.read_csv(prefix + 'fold7/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf8 = np.array(pd.read_csv(prefix + 'fold8/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])
Cscoresf9 = np.array(pd.read_csv(prefix + 'fold9/' + con + 'Cscore_integration_scaled2.csv').iloc[:, 1:8])

# %%
for k in range(10):
    results = pd.read_csv(prefix + 'fold' + str(k) + '/' + con + 'Cscore_integration_scaled2.csv')
    y_test = tf.keras.utils.to_categorical(results["Label"])
    y_score = np.array(results.iloc[:, 1:8])
    n_classes = 7

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # print(len(mean_tpr))
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    # plt.figure()
    lw = 2
    fig, ax = plt.subplots()
    ax.plot(fpr["micro"], tpr["micro"],
            label='Average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    class_label = ['breast', 'colorectum', 'liver', 'lung', 'ovary', 'pancreas', 'upper GI']
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of {0} cancer (area = {1:0.2f})'
                      ''.format(class_label[i], roc_auc[i]))

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='AUC of CancerTL')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='By Chance (area = 0.5)', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xlabel('False Positive Rate', size=16)
    plt.ylabel('True Positive Rate', size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    ax.legend(loc="lower right", prop={'size': 11})
    plt.savefig(prefix + 'viz/fold' + str(k) + 'AUC.pdf')
    plt.savefig(prefix + 'viz/fold' + str(k) + 'AUC.png')
    plt.show()
#%% DL
filename = "fold1_DL_AUC.csv"
DLf1 = np.array(pd.read_csv(filename).iloc[:, 1:8])

# %%
results = pd.read_csv(filename)
y_test = tf.keras.utils.to_categorical(results["Label"])
y_score = np.array(results.iloc[:, 1:8])
n_classes = 7

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# print(len(mean_tpr))
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
# plt.figure()
lw = 2
fig, ax = plt.subplots()
ax.plot(fpr["micro"], tpr["micro"],
label='Average ROC curve (area = {0:0.2f})'
      ''.format(roc_auc["micro"]),
color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
class_label = ['breast', 'colorectum', 'liver', 'lung', 'ovary', 'pancreas', 'upper GI']
for i, color in zip(range(n_classes), colors):
    ax.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of {0} cancer (area = {1:0.2f})'
                                                      ''.format(class_label[i], roc_auc[i]))

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='AUC of CancerDL')
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='By Chance (area = 0.5)', alpha=.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
plt.xlabel('False Positive Rate', size=16)
plt.ylabel('True Positive Rate', size=16)
plt.xticks(size=16)
plt.yticks(size=16)
ax.legend(loc="lower right", prop={'size': 11})
plt.savefig(prefix + 'viz/DL/fold1DLAUC.pdf')
plt.savefig(prefix + 'viz/DL/fold1DLAUC.png')
plt.show()

# %% Integrated cancerTL model performance  barplot
# TL and DL (make sure using the same dataset after spliting)

df = pd.read_csv(prefix + 'IntegratedCancerTL_AUC.csv')

# get the ACC AUC AP F1 MCC of test dataset results for CancerTL, CancerDL


# %%
# Draw a nested barplot by species and sex
# fig, ax = plt.subplots(figsize=(14, 8))
sns.set(font_scale=1.5, rc={"xtick.bottom": True, "ytick.left": True})
sns.set_style("white")
g = sns.catplot(data=df, kind="bar", x="Model", y="Value", palette="deep", height=6, aspect=8 / 6, errwidth=1.5,
                capsize=0.1)
# g.despine()
g.set_axis_labels("", "Measurements")
plt.title("Performance comparisons"
          ""
          " of CancerTL and others")
plt.ylim(0.75, 0.9)
plt.xticks(rotation=45, size=15)
plt.yticks(size=15)
plt.subplots_adjust(bottom=0.2, left=0.1)
plt.savefig(prefix + 'viz/Fig5B.pdf')
plt.show()
