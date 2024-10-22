import pandas as pd
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings("ignore")


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):

            pretty(value, indent + 2)
            print('-----------------------')
        else:
            print('\t' * (indent + 2) + str(value))
            print(" ")
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
def rms(arr):
    n = len(arr)
    squared = np.array(arr) * np.array(arr)
    sum = np.sum(squared)
    mean = (sum / (float)(n))
    root = math.sqrt(mean)
    return root
def mav(arr):
    n = len(arr)
    mav = sum((abs(np.array(arr))))/(float)(n)
    return mav
def var(arr):
    n = len(arr)
    squared = np.array(arr) * np.array(arr)
    sum = np.sum(squared)
    result = (sum / (float)(n))
    return result
def wl(arr):
    n = len(arr)
    sum = 0
    for i in range(1, n):
        sum += abs(arr[i] - arr[i - 1])
    return sum
def iav(arr):
    return sum(abs(np.array(arr)))

def extractSubject(name):
    ex1Path = 'DB4/' + name + '/' + name + '_E1_A1.mat'
    print(ex1Path)
    ex1 = scipy.io.loadmat(ex1Path)
    emg = ex1['emg']
    EMGdf = pd.DataFrame.from_dict(emg)
    stimulus = ex1['stimulus']

    ex2Path = 'DB4/' + name + '/' + name + '_E2_A1.mat'
    ex2 = scipy.io.loadmat(ex2Path)
    emg2 = ex2['emg']
    EMGdf2 = pd.DataFrame.from_dict(emg2)
    stimulus2 = ex2['stimulus']

    ex3Path = 'DB4/' + name + '/' + name + '_E3_A1.mat'
    ex3 = scipy.io.loadmat(ex3Path)
    emg3 = ex3['emg']
    EMGdf3 = pd.DataFrame.from_dict(emg3)
    stimulus3 = ex3['stimulus']

    Movements = {}
    for m in range(1, 53):
        if (m < 13):
            movementIndices = np.where(stimulus == m)[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf
        elif (m < 30):
            movementIndices = np.where(stimulus2 == (m - 12))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf2
        else:
            movementIndices = np.where(stimulus3 == (m - 29))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf3

        Electrodes = {}
        for e in range(1, 11):
            temp = {}
            for r in range(1, 7):
                startIndex = repetitions[r - 1][0]
                LastIndex = repetitions[r - 1][len(repetitions[r - 1]) - 1]
                df = EMG.iloc[startIndex:LastIndex, e - 1]
                df.reset_index(drop=True, inplace=True)
                narray = df.to_numpy(dtype=None, copy=False)
                temp["R{0}".format(r)] = narray
            Electrodes["Electrode{0}".format(e)] = temp
        Movements["Movement{0}".format(m)] = Electrodes
    return Movements
def Average(lst):
    return sum(lst) / len(lst)



subjects_accuracy = pd.DataFrame(columns={'Accuracy', 'Accuracy_Modified'})
pca_window = []
pca_movement = []
final_df = pd.DataFrame(columns={'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
                           'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
                           'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
                           'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
                           'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
                           'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
                           'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
                           'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
                           'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
                           'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10',
                           'Train','Movement'})

svc_sub = []
for s in range(1,11):
    subject = 'S' + str(s)
    dff = pd.DataFrame.from_dict(extractSubject(subject))
    df = pd.DataFrame(columns={'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
                                  'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
                                  'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
                                  'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
                                  'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
                                  'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
                                  'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
                                  'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
                                  'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
                                  'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10',
                                  'Train','Movement'})
    for e in range(1, 11):
        i = 0
        electrode = 'Electrode' + str(e)
        for m in range(1,53):
                M = dff['Movement'+str(m)][electrode]
                for r in range(1, 7):
                    rep = "R" + str(r)
                    if (r in [1, 3, 4, 6]):
                        train = 1
                    else:
                        train = 0
                    for x in range(0, len(M[rep]), 20):
                        df.at[i, 'RMS' + str(e)] = rms(M[rep][x:x + 50])
                        df.at[i, 'MAV' + str(e)] = mav(M[rep][x:x + 50])
                        df.at[i, 'VAR' + str(e)] = var(M[rep][x:x + 50])
                        df.at[i, 'WL' + str(e)] = wl(M[rep][x:x + 50])
                        df.at[i, 'IAV' + str(e)] = iav(M[rep][x:x + 50])
                        df.at[i, 'Movement'] = m
                        df.at[i, 'Train'] = train
                        i += 1
    final_df = final_df.append(df, ignore_index=True)
final_df.to_csv('df_new_4.csv')

#final_df = pd.read_csv('df_new_2.csv')
features = {'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
            'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
            'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
            'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
            'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
            'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
            'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
            'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
            'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
            'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10'}
# X_train = final_df[final_df['Train'] == 1].loc[:, features]
# scalar =  MinMaxScaler()
# scalar = scalar.fit(X_train)
# X_train = scalar.transform(X_train)
# X_test = final_df[final_df['Train'] == 0].loc[:, features]
# X_test = scalar.transform(X_test)
# y_train = final_df[final_df['Train'] == 1].loc[:, 'Movement']
# y_test = final_df[final_df['Train'] == 0].loc[:, 'Movement']
X_train = final_df[final_df['Train'] == 1].loc[:, features]
X_test = final_df[final_df['Train'] == 0].loc[:, features]
y_train = final_df[final_df['Train'] == 1]['Movement'].astype('int')
y_test = final_df[final_df['Train'] == 0]['Movement'].astype('int')
clf = RandomForestClassifier(n_estimators=100)
# clf = LinearDiscriminantAnalysis()
print('Sub')
clf.fit(X_train, y_train)
print('finished fitting')
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_test_new = [most_frequent(y_test[x:x + 26]) for x in range(0, len(y_test), 26)]
y_predicted_new = [most_frequent(y_pred[x:x + 26]) for x in range(0, len(y_pred), 26)]
accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
print("Window Accuracy",accuracy)
print("Movement Accuracy", accuracy_modified)

# cf_matrix = confusion_matrix(y_test, y_pred)

# print(cf_matrix)   

import seaborn as sns
import matplotlib.pyplot as plt

### Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

## Get Class Labels
# labels = le.classes_
class_names = range(1,53)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(14, 10))
ax= plt.subplot()
sns.heatmap(cm, annot=False, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()
# # final_df = pd.read_csv('df.csv')
# print(final_df)
# #final_df.to_csv('df.csv')
# lab_enc = preprocessing.LabelEncoder()
# features = {'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
#             'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
#             'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
#             'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
#             'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
#             'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
#             'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
#             'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
#             'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
#             'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10'}
# # x = final_df.loc[:, features]
# # y = final_df.loc[:,['Movement']].values
# # y=y.astype('int')
# # # x = StandardScaler().fit_transform(x)
# X_train = final_df[final_df['Train'] == 1].loc[:, features]
# X_test = final_df[final_df['Train'] == 0].loc[:, features]
# y_train = final_df[final_df['Train'] == 1]['Movement'].astype('int')
# y_test = final_df[final_df['Train'] == 0]['Movement'].astype('int')
# # # # pcas_acc=[]
# # # # for p in range(5,50):
# # # #     pca = PCA(n_components=p)
# # # #     principalComponents = pca.fit_transform(x)
# # # #     principalDf = pd.DataFrame(data=principalComponents)
# # # #     finalDf = pd.concat([principalDf, final_df['Movement'], final_df['Train']], axis=1)
# # # #     X_train = finalDf[finalDf['Train'] == 1]
# # # #     X_train.drop({'Movement', 'Train'}, axis=1, inplace=True)
# # # #     X_test = finalDf[finalDf['Train'] == 0]
# # # #     X_test.drop({'Movement', 'Train'}, axis=1, inplace=True)
# # # #     y_train = finalDf[finalDf['Train'] == 1]['Movement'].astype('int')
# # # #     y_test = finalDf[finalDf['Train'] == 0]['Movement'].astype('int')
# # # #
# # # #     model = KNeighborsClassifier(n_neighbors=1)
# # # #     model.fit(X_train,y_train)
# # # #     pred_i = model.predict(X_test)
# # # #     accuracy = accuracy_score(y_test, pred_i)
# # # #     y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
# # # #     y_predicted_new = [most_frequent(pred_i[x:x + 11]) for x in range(0, len(pred_i), 11)]
# # # #     accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
# # # #     print("Window Accuracy",accuracy)
# # # #     print("Movement Accuracy", accuracy_modified)
# # # #     pcas_acc.append(accuracy_modified)
# # # #
# # # # plt.figure(figsize=(10,6))
# # # # plt.plot(range(5,50),pcas_acc,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)
# # # # plt.title('Accuracy vs. PCA components')
# # # # plt.xlabel('PCA components')
# # # # plt.ylabel('Accuracy')
# # # # plt.show()


# # clf = RandomForestClassifier()
# # clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
# # y_predicted_new = [most_frequent(y_pred[x:x + 11]) for x in range(0, len(y_pred), 11)]
# # accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
# #
# # print("Window Accuracy",accuracy)
# # print("Movement Accuracy", accuracy_modified)

# # print(confusion_matrix(y_test_new, y_predicted_new))
# # # Printing the precision and recall, among other metrics
# # print(classification_report(y_test_new, y_predicted_new))
# #
# # #Get the confusion matrix
# # cf_matrix = confusion_matrix(y_test_new, y_predicted_new)
# # ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
# #
# # ax.set_title('Movements Classification Confusion Matrix\n');
# # ax.set_xlabel('\nPredicted Movement')
# # ax.set_ylabel('Actual Movement ');
# #
# # ## Ticket labels - List must be in alphabetical order
# # ax.set_xticks(range(1,51))
# # ax.set_yticks(range(1,51))
# #
# # ax.xaxis.set_ticklabels(range(1,51))
# # ax.yaxis.set_ticklabels(range(1,51))
# # ## Display the visualization of the Confusion Matrix.
# # plt.show()

# # pcas =[0.44, 0.56, 0.64, 0.68, 0.73, 0.76, 0.77, 0.77, 0.78, 0.79, 0.79, 0.79, 0.79, 0.8, 0.8, 0.8, 0.79, 0.8, 0.81, 0.79, 0.81, 0.79, 0.8, 0.8, 0.816, 0.8, 0.81, 0.81, 0.81, 0.81, 0.81, 0.79, 0.81, 0.81, 0.79, 0.81, 0.8, 0.79, 0.8, 0.8, 0.8, 0.79, 0.8, 0.79, 0.79, 0.8]
# # # for p in range(5,51):
# # #     pca = PCA(n_components=p)
# # #     principalComponents = pca.fit_transform(x)
# # #     principalDf = pd.DataFrame(data=principalComponents)
# # #     finalDf = pd.concat([principalDf, final_df['Movement'], final_df['Train']], axis=1)
# # #
# # #     X_train = finalDf[finalDf['Train'] == 1]
# # #     X_train.drop({'Movement', 'Train'}, axis=1, inplace=True)
# # #     X_test = finalDf[finalDf['Train'] == 0]
# # #     X_test.drop({'Movement', 'Train'}, axis=1, inplace=True)
# # #     y_train = finalDf[finalDf['Train'] == 1]['Movement'].astype('int')
# # #     y_test = finalDf[finalDf['Train'] == 0]['Movement'].astype('int')
# # #
# # #     clf = RandomForestClassifier()
# # #     clf.fit(X_train, y_train)
# # #     y_pred = clf.predict(X_test)
# # #     accuracy = accuracy_score(y_test, y_pred)
# # #     y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
# # #     y_predicted_new = [most_frequent(y_pred[x:x + 11]) for x in range(0, len(y_pred), 11)]
# # #     accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
# # #
# # #     print("Window Accuracy",accuracy)
# # #     print("Movement Accuracy", accuracy_modified)
# # #     pcas.append(accuracy_modified)
# # print(pcas)
# # pca_window.append(accuracy)
# # pca_movement.append(accuracy_modified)
# # print(pcas)

# #subjects = ['S1', 'S2', 'S3']
# # subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
# #             'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
# #             'S2l)
# # x = range(5,51)  # the label locations
# # print(x)
# # width = 0.1  # the width of the bars
# # # # #
# # fig, ax = plt.subplots()
# # # #window = ax.bar(x - width/2, window, width, label='Window Accuracy')
# # movement = ax.bar(x, pcas, width, label='Movement Accuracy')
# #
# # ax.set_ylabel('Accuracy')
# # ax.set_xlabel('PCA')
# # ax.set_title('PCAs Movement Accuracies')
# # # #ax.set_title('Average Subject Accuracies Per PCA')
# # ax.set_xticks(x, range(5,51))
# # ax.legend()
# # #ax.bar_label(window)
# #ax.bar_label(pcas)
# # fig.tight_layout()
# plt.show()

# # pca = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
# # w = [0.45, 0.67, 0.76, 0.82, 0.86, 0.86, 0.88, 0.89, 0.89, 0.89, 0.89]
# # m = [0.46, 0.76, 0.86, 0.93, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.96]
# # x = np.arange(len(pca))  # the label locations
# # width = 0.2  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # w = ax.bar(x - width/2, w, width, label='Average Window Accuracy')
# # m = ax.bar(x + width/2, m, width, label='Average Movement Accuracy')
# #
# # ax.set_ylabel('Accuracy')
# # ax.set_xlabel('PCA')
# # ax.set_title('PCA Accuracies')
# # ax.set_xticks(x, pca)
# # ax.legend()
# # ax.bar_label(w, padding=3)
# # ax.bar_label(m, padding=3)
# # fig.tight_layout()
# # plt.show()
# # electrode1_acc = []; electrode1_mod = []; electrode2_acc = []; electrode2_mod = []; electrode3_acc = []; electrode3_mod = []; electrode4_acc = [];
# # electrode4_mod = []; electrode5_acc = []; electrode5_mod = []; electrode6_acc = []; electrode6_mod = []; electrode7_acc = []; electrode7_mod = [];
# # electrode8_acc = []; electrode8_mod = []; electrode9_acc = []; electrode9_mod = []; electrode10_acc = []; electrode10_mod = []
# #
# # for d in Subjects_Accuracies.values():
# #     electrode1_acc.append(round(d['Electrode1']['Accuracy'],2))
# #     electrode1_mod.append(d['Electrode1']['Accuracy_Modified'])
# #
# #     electrode2_acc.append(round(d['Electrode2']['Accuracy'],2))
# #     electrode2_mod.append(d['Electrode2']['Accuracy_Modified'])
# #
# #     electrode3_acc.append(round(d['Electrode3']['Accuracy'],2))
# #     electrode3_mod.append(d['Electrode3']['Accuracy_Modified'])
# #
# #     electrode4_acc.append(round(d['Electrode4']['Accuracy'],2))
# #     electrode4_mod.append(d['Electrode4']['Accuracy_Modified'])
# #
# #     electrode5_acc.append(d['Electrode5']['Accuracy'])
# #     electrode5_mod.append(d['Electrode5']['Accuracy_Modified'])
# #
# #     electrode6_acc.append(d['Electrode6']['Accuracy'])
# #     electrode6_mod.append(d['Electrode6']['Accuracy_Modified'])
# #
# #     electrode7_acc.append(d['Electrode7']['Accuracy'])
# #     electrode7_mod.append(d['Electrode7']['Accuracy_Modified'])
# #
# #     electrode8_acc.append(d['Electrode8']['Accuracy'])
# #     electrode8_mod.append(d['Electrode8']['Accuracy_Modified'])
# #
# #     electrode9_acc.append(d['Electrode9']['Accuracy'])
# #     electrode9_mod.append(d['Electrode9']['Accuracy_Modified'])
# #
# #     electrode10_acc.append(d['Electrode10']['Accuracy'])
# #     electrode10_mod.append(d['Electrode10']['Accuracy_Modified'])

# # #############################################################################
# # Visualization
# #
# # draw visualization of parameter effects
# #
# # C_range = np.logspace(-2, 10, 13)
# # gamma_range = np.logspace(-9, 3, 13)
# # param_grid = dict(gamma=gamma_range, C=C_range)
# # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# # grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# # grid.fit(X, y)
# # scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))
# #
# # # Draw heatmap of the validation accuracy as a function of gamma and C
# # #
# # # The score are encoded as colors with the hot colormap which varies from dark
# # # red to bright yellow. As the most interesting scores are all located in the
# # # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# # # as to make it easier to visualize the small variations of score values in the
# # # interesting range while not brutally collapsing all the low score values to
# # # the same color.
# # #
# # plt.figure(figsize=(8, 6))
# # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
# # plt.imshow(
# #     scores,
# #     interpolation="nearest",
# #     cmap=plt.cm.hot,
# #     norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
# # )
# # plt.xlabel("gamma")
# # plt.ylabel("C")
# # plt.colorbar()
# # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# # plt.yticks(np.arange(len(C_range)), C_range)
# # plt.title("Validation accuracy")
# # plt.show()

# # subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
# #
# # x = np.arange(len(subjects))  # the label locations
# # width = 0.08  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - 9*width/2, electrode1_acc, width, label='Electrode 1')
# # rects2 = ax.bar(x - 7*width/2, electrode2_acc, width, label='Electrode 2')
# # rects3 = ax.bar(x - 5*width/2, electrode3_acc, width, label='Electrode 3')
# # rects4 = ax.bar(x - 3*width/2, electrode4_acc, width, label='Electrode 4')
# # rects5 = ax.bar(x - width/2, electrode5_acc, width, label='Electrode 5')
# # rects6 = ax.bar(x + width/2, electrode6_acc, width, label='Electrode 6')
# # rects7 = ax.bar(x + 3*width/2, electrode7_acc, width, label='Electrode 7')
# # rects8 = ax.bar(x + 5*width/2, electrode8_acc, width, label='Electrode 8')
# # rects9 = ax.bar(x + 7*width/2, electrode9_acc, width, label='Electrode 9')
# # rects10 = ax.bar(x + 9*width/2, electrode10_acc, width, label='Electrode 10')
# #
# # ax.set_ylabel('Accuracy')
# # ax.set_title('Window Accuracy')
# # ax.set_xticks(x, subjects)
# # ax.legend()
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# # ax.bar_label(rects3, padding=3)
# # ax.bar_label(rects4, padding=3)
# # fig.tight_layout()
# # plt.show()


# # subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
# #
# # x = np.arange(len(subjects))  # the label locations
# # width = 0.08  # the width of the bars
# #
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - 9*width/2, electrode1_mod, width, label='Electrode 1')
# # rects2 = ax.bar(x - 7*width/2, electrode2_mod, width, label='Electrode 2')
# # rects3 = ax.bar(x - 5*width/2, electrode3_mod, width, label='Electrode 3')
# # rects4 = ax.bar(x - 3*width/2, electrode4_mod, width, label='Electrode 4')
# # rects5 = ax.bar(x - width/2, electrode5_mod, width, label='Electrode 5')
# # rects6 = ax.bar(x + width/2, electrode6_mod, width, label='Electrode 6')
# # rects7 = ax.bar(x + 3*width/2, electrode7_mod, width, label='Electrode 7')
# # rects8 = ax.bar(x + 5*width/2, electrode8_mod, width, label='Electrode 8')
# # rects9 = ax.bar(x + 7*width/2, electrode9_mod, width, label='Electrode 9')
# # rects10 = ax.bar(x + 9*width/2, electrode10_mod, width, label='Electrode 10')
# #
# # ax.set_ylabel('Accuracy')
# # ax.set_title('Movement Accuracy')
# # ax.set_xticks(x, subjects)
# # ax.legend()
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# # ax.bar_label(rects3, padding=3)
# # ax.bar_label(rects4, padding=3)
# # fig.tight_layout()
# # plt.show()

# #
# # X_train['Movement'] = y_train
# # df= X_train
# # fig, (ax1, ax2) = plt.subplots(1,2)
# # fig.suptitle('Electrode 1')
# # ax1.set_title('Movement 1')
# # ax1.plot(df.where(df['Movement'] == 0))
# #
# # ax2.set_title('Movement 11')
# # ax2.plot(df.where(df['Movement'] == 1))
# #
# # # ax3.set_title('Movement 28')
# # # ax3.plot(df.where(df['Movement'] == 2))
# #
# # plt.show()

# # df = pd.DataFrame.from_dict(electrodes)
# # E1M1 = pd.DataFrame.from_dict(df['Electrode1']['Movement1'])
# # # E1M2 = pd.DataFrame.from_dict(df['Electrode1']['Movement2'])
# # # E1M3 = pd.DataFrame.from_dict(df['Electrode1']['Movement3'])
# # # E1M4 = pd.DataFrame.from_dict(df['Electrode1']['Movement4'])
# # # E1M5 = pd.DataFrame.from_dict(df['Electrode1']['Movement5'])
# # # E1M6 = pd.DataFrame.from_dict(df['Electrode1']['Movement6'])
# #
# #
# # print("---------------Before Scaling-------------------------------")
# # stats_df = X_train.describe()
# # stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# # # out_fields = ['mean','25%','50%','75%', 'range']
# # # stats_df = stats_df.loc[out_fields]
# # stats_df.rename({'50%': 'median'}, inplace=True)
# # print(stats_df)
# # print("----------------------------------------------")
# #
# # print("----------------After Scaling------------------------------")
# # stats_df = scaled_X_train.describe()
# # stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# # # out_fields = ['mean','25%','50%','75%', 'range']
# # # stats_df = stats_df.loc[out_fields]
# # stats_df.rename({'50%': 'median'}, inplace=True)
# # print(stats_df)
# # print("----------------------------------------------")
# # print(E1M1.skew().sort_values(ascending=False))
# #
# # # ax = plt.axes()
# # #
# # # ax.scatter(E1M1.IAV, E1M1.MAV)
# # #
# # # # Label the axes
# # # ax.set(xlabel='IAV  (cm)',
# # #        ylabel='MAV  (cm)',
# # #        title='IAV vs MAV');
# # #
# # # plt.show()
# #
# # # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# # # fig.suptitle('RMS')
# # #
# # # ax1.set_title('Electrode 1 Movement 1')
# # # ax1.plot(E1M1['RMS'])
# # #
# # # ax2.set_title('Electrode 1 Movement 2 ')
# # # ax2.plot(E1M2['RMS'])
# # #
# # # ax3.set_title('Electrode 1 Movement 3 ')
# # # ax3.plot(E1M3['RMS'])
# # #
# # # ax4.set_title('Electrode 1 Movement 4 ')
# # # ax4.plot(E1M4['RMS'])
# # #
# # # ax5.set_title('Electrode 1 Movement 5 ')
# # # ax5.plot(E1M5['RMS'])
# # #
# # # ax6.set_title('Electrode 1 Movement 6 ')
# # # ax6.plot(E1M6['RMS'])
# # #
# # # plt.show()
# #
# #
# #
