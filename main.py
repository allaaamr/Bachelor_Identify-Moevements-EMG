import pandas as pd
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from collections import Counter
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
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
    square = 0
    for i in range(0, n):
        square += (arr[i] ** 2)
    mean = (square / (float)(n))
    root = math.sqrt(mean)
    return root
def mav(arr):
    n = len(arr)
    absSum = 0
    for i in range(0, n):
        absSum += abs(arr[i])
    mav = (absSum / (float)(n))
    return mav
def var(arr):
    n = len(arr)
    square = 0
    for i in range(0, n):
        square += (arr[i] ** 2)

    result = (square / (float)(n))
    return result
def wl(arr):
    n = len(arr)
    sum = 0
    for i in range(1, n):
        sum += abs(arr[i] - arr[i - 1])
    return sum
def iav(arr):
    n = len(arr)
    absSum = 0
    for i in range(0, n):
        absSum += abs(arr[i])
    return absSum
def extractSubject(name):
    ex1Path = 'DB1/' + name + '/' + name + '_A1_E1.mat'
    print(ex1Path)
    ex1 = scipy.io.loadmat(ex1Path)
    emg = ex1['emg']
    EMGdf = pd.DataFrame.from_dict(emg)
    stimulus = ex1['stimulus']

    ex2Path = 'DB1/' + name + '/' + name + '_A1_E2.mat'
    ex2 = scipy.io.loadmat(ex2Path)
    emg2 = ex2['emg']
    EMGdf2 = pd.DataFrame.from_dict(emg2)
    stimulus2 = ex2['stimulus']

    ex3Path = 'DB1/' + name + '/' + name + '_A1_E3.mat'
    ex3 = scipy.io.loadmat(ex3Path)
    emg3 = ex3['emg']
    EMGdf3 = pd.DataFrame.from_dict(emg3)
    stimulus3 = ex3['stimulus']

    Movements = {}
    for m in range(1, 51):
        if (m < 11):
            movementIndices = np.where(stimulus == m)[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf
        elif (m < 28):
            movementIndices = np.where(stimulus2 == (m - 10))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf2
        else:
            movementIndices = np.where(stimulus3 == (m - 27))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf3

        Electrodes = {}
        for e in range(1, 11):
            temp = {}
            for r in range(1, 11):
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


for s in range(1,28):
    subject = "S" + str(s)
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
        for m in range(1,51):
            M = dff['Movement'+str(m)][electrode]
            for r in range(1, 7):
                rep = "R" + str(r)
                if (r in [1, 3, 4, 6]):
                    train = 1
                else:
                    train = 0
                data = {}; dataM11 = {}; dataM29 = {}
                for x in range(0, len(M[rep]), 48):
                    df.at[i, 'RMS' + str(e)] = rms(M[rep][x:x + 50])
                    df.at[i, 'MAV' + str(e)] = mav(M[rep][x:x + 50])
                    df.at[i, 'VAR' + str(e)] = var(M[rep][x:x + 50])
                    df.at[i, 'WL' + str(e)] = wl(M[rep][x:x + 50])
                    df.at[i, 'IAV' + str(e)] = iav(M[rep][x:x + 50])
                    df.at[i, 'Movement'] = m
                    df.at[i, 'Train'] = train
                    i += 1

    final_df = final_df.append(df, ignore_index=True)

for p in range(4,18):
    lab_enc = preprocessing.LabelEncoder()
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

    x = final_df.loc[:, features].values
    y = final_df.loc[:,['Movement']].values
    y=y.astype('int')
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=p)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, df['Movement'], df['Train']], axis=1)

    X_train = finalDf[finalDf['Train'] == 1]
    X_train.drop({'Movement', 'Train'}, axis=1, inplace=True)
    X_test = finalDf[finalDf['Train'] == 0]
    X_test.drop({'Movement', 'Train'}, axis=1, inplace=True)
    y_train = finalDf[finalDf['Train'] == 1]['Movement'].astype('int')
    y_test = finalDf[finalDf['Train'] == 0]['Movement'].astype('int')

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
    y_predicted_new = [most_frequent(y_pred[x:x + 11]) for x in range(0, len(y_pred), 11)]
    accuracy_modified = accuracy_score(y_test_new, y_predicted_new)

    print(p)
    print("Window Accuracy",accuracy)
    print("Movement Accuracy", accuracy_modified)
    pca_window.append(accuracy)
    pca_movement.append(accuracy_modified)

    # print(Average(window))
    # print(Average(movement))

#subjects = ['S1', 'S2', 'S3']
# subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
#             'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
#             'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27']
pca = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
x = np.arange(len(pca))  # the label locations
width = 0.1  # the width of the bars
# #
fig, ax = plt.subplots()
#window = ax.bar(x - width/2, window, width, label='Window Accuracy')
movement = ax.bar(x, pca_movement, width, label='Movement Accuracy')

ax.set_ylabel('Accuracy')
ax.set_xlabel('PCA')
ax.set_title('PCAs Movement Accuracies')
#ax.set_title('Average Subject Accuracies Per PCA')
ax.set_xticks(x, pca)
ax.legend()
#ax.bar_label(window)
ax.bar_label(movement)
fig.tight_layout()
plt.show()

# pca = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
# w = [0.45, 0.67, 0.76, 0.82, 0.86, 0.86, 0.88, 0.89, 0.89, 0.89, 0.89]
# m = [0.46, 0.76, 0.86, 0.93, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.96]
# x = np.arange(len(pca))  # the label locations
# width = 0.2  # the width of the bars
#
# fig, ax = plt.subplots()
# w = ax.bar(x - width/2, w, width, label='Average Window Accuracy')
# m = ax.bar(x + width/2, m, width, label='Average Movement Accuracy')
#
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('PCA')
# ax.set_title('PCA Accuracies')
# ax.set_xticks(x, pca)
# ax.legend()
# ax.bar_label(w, padding=3)
# ax.bar_label(m, padding=3)
# fig.tight_layout()
# plt.show()
# electrode1_acc = []; electrode1_mod = []; electrode2_acc = []; electrode2_mod = []; electrode3_acc = []; electrode3_mod = []; electrode4_acc = [];
# electrode4_mod = []; electrode5_acc = []; electrode5_mod = []; electrode6_acc = []; electrode6_mod = []; electrode7_acc = []; electrode7_mod = [];
# electrode8_acc = []; electrode8_mod = []; electrode9_acc = []; electrode9_mod = []; electrode10_acc = []; electrode10_mod = []
#
# for d in Subjects_Accuracies.values():
#     electrode1_acc.append(round(d['Electrode1']['Accuracy'],2))
#     electrode1_mod.append(d['Electrode1']['Accuracy_Modified'])
#
#     electrode2_acc.append(round(d['Electrode2']['Accuracy'],2))
#     electrode2_mod.append(d['Electrode2']['Accuracy_Modified'])
#
#     electrode3_acc.append(round(d['Electrode3']['Accuracy'],2))
#     electrode3_mod.append(d['Electrode3']['Accuracy_Modified'])
#
#     electrode4_acc.append(round(d['Electrode4']['Accuracy'],2))
#     electrode4_mod.append(d['Electrode4']['Accuracy_Modified'])
#
#     electrode5_acc.append(d['Electrode5']['Accuracy'])
#     electrode5_mod.append(d['Electrode5']['Accuracy_Modified'])
#
#     electrode6_acc.append(d['Electrode6']['Accuracy'])
#     electrode6_mod.append(d['Electrode6']['Accuracy_Modified'])
#
#     electrode7_acc.append(d['Electrode7']['Accuracy'])
#     electrode7_mod.append(d['Electrode7']['Accuracy_Modified'])
#
#     electrode8_acc.append(d['Electrode8']['Accuracy'])
#     electrode8_mod.append(d['Electrode8']['Accuracy_Modified'])
#
#     electrode9_acc.append(d['Electrode9']['Accuracy'])
#     electrode9_mod.append(d['Electrode9']['Accuracy_Modified'])
#
#     electrode10_acc.append(d['Electrode10']['Accuracy'])
#     electrode10_mod.append(d['Electrode10']['Accuracy_Modified'])

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

# scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
#
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(
#     scores,
#     interpolation="nearest",
#     cmap=plt.cm.hot,
#     norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
# )
# plt.xlabel("gamma")
# plt.ylabel("C")
# plt.colorbar()
# plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.title("Validation accuracy")
# plt.show()

# subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
#
# x = np.arange(len(subjects))  # the label locations
# width = 0.08  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 9*width/2, electrode1_acc, width, label='Electrode 1')
# rects2 = ax.bar(x - 7*width/2, electrode2_acc, width, label='Electrode 2')
# rects3 = ax.bar(x - 5*width/2, electrode3_acc, width, label='Electrode 3')
# rects4 = ax.bar(x - 3*width/2, electrode4_acc, width, label='Electrode 4')
# rects5 = ax.bar(x - width/2, electrode5_acc, width, label='Electrode 5')
# rects6 = ax.bar(x + width/2, electrode6_acc, width, label='Electrode 6')
# rects7 = ax.bar(x + 3*width/2, electrode7_acc, width, label='Electrode 7')
# rects8 = ax.bar(x + 5*width/2, electrode8_acc, width, label='Electrode 8')
# rects9 = ax.bar(x + 7*width/2, electrode9_acc, width, label='Electrode 9')
# rects10 = ax.bar(x + 9*width/2, electrode10_acc, width, label='Electrode 10')
#
# ax.set_ylabel('Accuracy')
# ax.set_title('Window Accuracy')
# ax.set_xticks(x, subjects)
# ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
# ax.bar_label(rects4, padding=3)
# fig.tight_layout()
# plt.show()


# subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
#
# x = np.arange(len(subjects))  # the label locations
# width = 0.08  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - 9*width/2, electrode1_mod, width, label='Electrode 1')
# rects2 = ax.bar(x - 7*width/2, electrode2_mod, width, label='Electrode 2')
# rects3 = ax.bar(x - 5*width/2, electrode3_mod, width, label='Electrode 3')
# rects4 = ax.bar(x - 3*width/2, electrode4_mod, width, label='Electrode 4')
# rects5 = ax.bar(x - width/2, electrode5_mod, width, label='Electrode 5')
# rects6 = ax.bar(x + width/2, electrode6_mod, width, label='Electrode 6')
# rects7 = ax.bar(x + 3*width/2, electrode7_mod, width, label='Electrode 7')
# rects8 = ax.bar(x + 5*width/2, electrode8_mod, width, label='Electrode 8')
# rects9 = ax.bar(x + 7*width/2, electrode9_mod, width, label='Electrode 9')
# rects10 = ax.bar(x + 9*width/2, electrode10_mod, width, label='Electrode 10')
#
# ax.set_ylabel('Accuracy')
# ax.set_title('Movement Accuracy')
# ax.set_xticks(x, subjects)
# ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
# ax.bar_label(rects4, padding=3)
# fig.tight_layout()
# plt.show()

#
# X_train['Movement'] = y_train
# df= X_train
# fig, (ax1, ax2) = plt.subplots(1,2)
# fig.suptitle('Electrode 1')
# ax1.set_title('Movement 1')
# ax1.plot(df.where(df['Movement'] == 0))
#
# ax2.set_title('Movement 11')
# ax2.plot(df.where(df['Movement'] == 1))
#
# # ax3.set_title('Movement 28')
# # ax3.plot(df.where(df['Movement'] == 2))
#
# plt.show()

# df = pd.DataFrame.from_dict(electrodes)
# E1M1 = pd.DataFrame.from_dict(df['Electrode1']['Movement1'])
# # E1M2 = pd.DataFrame.from_dict(df['Electrode1']['Movement2'])
# # E1M3 = pd.DataFrame.from_dict(df['Electrode1']['Movement3'])
# # E1M4 = pd.DataFrame.from_dict(df['Electrode1']['Movement4'])
# # E1M5 = pd.DataFrame.from_dict(df['Electrode1']['Movement5'])
# # E1M6 = pd.DataFrame.from_dict(df['Electrode1']['Movement6'])
#
#
# print("---------------Before Scaling-------------------------------")
# stats_df = X_train.describe()
# stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# # out_fields = ['mean','25%','50%','75%', 'range']
# # stats_df = stats_df.loc[out_fields]
# stats_df.rename({'50%': 'median'}, inplace=True)
# print(stats_df)
# print("----------------------------------------------")
#
# print("----------------After Scaling------------------------------")
# stats_df = scaled_X_train.describe()
# stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# # out_fields = ['mean','25%','50%','75%', 'range']
# # stats_df = stats_df.loc[out_fields]
# stats_df.rename({'50%': 'median'}, inplace=True)
# print(stats_df)
# print("----------------------------------------------")
# print(E1M1.skew().sort_values(ascending=False))
#
# # ax = plt.axes()
# #
# # ax.scatter(E1M1.IAV, E1M1.MAV)
# #
# # # Label the axes
# # ax.set(xlabel='IAV  (cm)',
# #        ylabel='MAV  (cm)',
# #        title='IAV vs MAV');
# #
# # plt.show()
#
# # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# # fig.suptitle('RMS')
# #
# # ax1.set_title('Electrode 1 Movement 1')
# # ax1.plot(E1M1['RMS'])
# #
# # ax2.set_title('Electrode 1 Movement 2 ')
# # ax2.plot(E1M2['RMS'])
# #
# # ax3.set_title('Electrode 1 Movement 3 ')
# # ax3.plot(E1M3['RMS'])
# #
# # ax4.set_title('Electrode 1 Movement 4 ')
# # ax4.plot(E1M4['RMS'])
# #
# # ax5.set_title('Electrode 1 Movement 5 ')
# # ax5.plot(E1M5['RMS'])
# #
# # ax6.set_title('Electrode 1 Movement 6 ')
# # ax6.plot(E1M6['RMS'])
# #
# # plt.show()
#
#
#
