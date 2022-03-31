import pandas as pd
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import MaxAbsScaler

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
    sum=0
    for i in range(1, n):
        sum += abs(arr[i]-arr[i-1])
    return sum
def iav(arr):
    n = len(arr)
    absSum = 0
    for i in range(0, n):
        absSum += abs(arr[i])
    return absSum


def extractSubject (name):
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
        # looping over the 12 electrodes of each movement
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

#
#
# # Number of samples in normalized_tone
#
# # print(dff.loc['Electrode1']['Movement1'])
# # fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4)
# # fig.suptitle('Electrode 1 Movement 2')
# #
# # ax1.set_title('R1')
# # ax1.plot(dff.loc['Electrode1']['Movement2']['R1'])
# #
# # ax2.set_title('R2')
# # ax2.plot(dff.loc['Electrode1']['Movement2']['R2'])
# #
# # ax3.set_title('R3')
# # ax3.plot(dff.loc['Electrode1']['Movement2']['R3'])
# #
# # ax4.set_title('R4')
# # ax4.plot(dff.loc['Electrode1']['Movement2']['R4'])
# #
# # ax5.set_title('R5')
# # ax5.plot(dff.loc['Electrode1']['Movement2']['R5'])
# #
# # ax6.set_title('R6')
# # ax6.plot(dff.loc['Electrode1']['Movement2']['R6'])
# #
# # ax7.set_title('R7')
# # ax7.plot(dff.loc['Electrode1']['Movement2']['R7'])
# #
# # ax8.set_title('R8')
# # ax8.plot(dff.loc['Electrode1']['Movement2']['R8'])
# # plt.show()
#


## Features

# train = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})
# test = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})
#
# for r in range(1, 7):
#     rep = "R" + str(r)
#     if(r in [1, 3, 4, 6]):
#         df = train
#     else:
#         df = test
#
#     for x in range(0, len( M1E1[rep]), 48):
#         rms_value = rms(M1E1[rep][x:x + 50])
#         mav_value = mav(M1E1[rep][x:x + 50])
#         var_value = var(M1E1[rep][x:x + 50])
#         wl_value = wl(M1E1[rep][x:x + 50])
#         iav_value= iav(M1E1[rep][x:x + 50])
#         movement = 0
#         df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value,'Movement':movement}
#
#     for x in range(0, len( M11E1[rep]), 48):
#         rms_value = rms(M11E1[rep][x:x + 50])
#         mav_value = mav(M11E1[rep][x:x + 50])
#         var_value = var(M11E1[rep][x:x + 50])
#         wl_value = wl(M11E1[rep][x:x + 50])
#         iav_value= iav(M11E1[rep][x:x + 50])
#         movement = 1
#         df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value,'Movement':movement}
#
#

subjects_accuracy = pd.DataFrame(columns= {'Accuracy', 'Accuracy_Modified'})


for i in range (1,11):
    subject = "S" +str(i)
    dff = pd.DataFrame.from_dict(extractSubject(subject))
    M1E1 = dff['Movement1']['Electrode1']
    M11E1 = dff['Movement11']['Electrode1']
    M1E2 = dff['Movement1']['Electrode2']
    M11E2 = dff['Movement11']['Electrode2']
    train = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2', 'Movement'})
    test = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2', 'Movement'})
    for r in range(1, 7):
        rep = "R" + str(r)
        if(r in [1, 3, 4, 6]):
            df = train
        else:
            df = test

        for x in range(0, len( M1E1[rep]), 48):
            rms_value = rms(M1E1[rep][x:x + 50])
            mav_value = mav(M1E1[rep][x:x + 50])
            var_value = var(M1E1[rep][x:x + 50])
            wl_value = wl(M1E1[rep][x:x + 50])
            iav_value= iav(M1E1[rep][x:x + 50])
            rms2_value = rms(M1E2[rep][x:x + 50])
            mav2_value = mav(M1E2[rep][x:x + 50])
            var2_value = var(M1E2[rep][x:x + 50])
            wl2_value = wl(M1E2[rep][x:x + 50])
            iav2_value= iav(M1E2[rep][x:x + 50])
            movement = 0
            df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value, 'RMS2':rms2_value, 'MAV2': mav2_value, 'VAR2':var2_value, 'WL2':wl2_value, "IAV2":iav2_value,'Movement':movement}

        for x in range(0, len( M11E1[rep]), 48):
            rms_value = rms(M11E1[rep][x:x + 50])
            mav_value = mav(M11E1[rep][x:x + 50])
            var_value = var(M11E1[rep][x:x + 50])
            wl_value = wl(M11E1[rep][x:x + 50])
            iav_value= iav(M11E1[rep][x:x + 50])
            rms2_value = rms(M11E2[rep][x:x + 50])
            mav2_value = mav(M11E2[rep][x:x + 50])
            var2_value = var(M11E2[rep][x:x + 50])
            wl2_value = wl(M11E2[rep][x:x + 50])
            iav2_value= iav(M11E2[rep][x:x + 50])
            movement = 1
            df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value, 'RMS2':rms2_value, 'MAV2': mav2_value, 'VAR2':var2_value, 'WL2':wl2_value, "IAV2":iav2_value,'Movement':movement}

        # for x in range(0, len( M28E1[rep]), 48):
        #     rms_value = rms(M28E1[rep][x:x + 50])
        #     mav_value = mav(M28E1[rep][x:x + 50])
        #     var_value = var(M28E1[rep][x:x + 50])
        #     wl_value = wl(M28E1[rep][x:x + 50])
        #     iav_value= iav(M28E1[rep][x:x + 50])
        #     rms2_value = rms(M28E2[rep][x:x + 50])
        #     mav2_value = mav(M28E2[rep][x:x + 50])
        #     var2_value = var(M28E2[rep][x:x + 50])
        #     wl2_value = wl(M28E2[rep][x:x + 50])
        #     iav2_value= iav(M28E2[rep][x:x + 50])
        #     movement = 2
        #     df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value, 'RMS2':rms2_value, 'MAV2': mav2_value, 'VAR2':var2_value, 'WL2':wl2_value, "IAV2":iav2_value,'Movement':movement}
    lab_enc = preprocessing.LabelEncoder()
    features = {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2'}
    X_train = pd.DataFrame.from_dict({k: train[k] for k in features})
    X_test = pd.DataFrame.from_dict({k: test[k] for k in features})
    y_train = lab_enc.fit_transform(train['Movement'])
    y_test = lab_enc.fit_transform(test['Movement'])

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:  " ,accuracy)
    y_test_new = [most_frequent(y_test[x:x + 11])  for x in range(0, len(y_test), 11)]
    y_predicted_new = [most_frequent(y_pred[x:x + 11])  for x in range(0, len(y_pred), 11)]
    accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
    # print("Accuracy Modified:  ", accuracy_modified)

    subjects_accuracy.loc[subjects_accuracy.shape[0]] = [accuracy, accuracy_modified]

print(subjects_accuracy)

# fig, axs = plt.subplots(1,2)
#
# axs.hist( subjects_accuracy['Accuracy'], bins = 10)
# plt.grid(axis='x', alpha=0.75)
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.set_xticks([1,2,3,4,5,6,7,8,9,10])
# ax1.set_xlabel('Subjects')
# ax1.set_title('Window Accuracy')
# dout= subjects_accuracy['Accuracy']
# ax1.hist(dout)
# ax1.set_title('Subjects')
# ax2.set_title('Movement Accuracy')
# ax2.hist(subjects_accuracy['Accuracy_Modified'], bins=10)
# ax2.set_xticks([1,2,3,4,5,6,7,8,9,10])
# ax2.set_xlabel('Subjects')
#
# plt.show()
d = subjects_accuracy['Accuracy'].to_dict()
print(d)
dic = {
    0 : 0.659091 ,
    1 : 0.909091 ,
    2 : 0.750000 ,
    3 : 0.931818  ,
    4 : 0.840909 ,
    5 : 0.772727 ,
    6 : 0.909091,
    7 : 0.704545   ,
    8 : 1.000000   ,
    9 : 0.727273,
}
fig, axs = plt.subplots(1,2)
plt.bar(d.keys(), d.values(), 2.0, color='b')
axs.set_xlabel('Subjects')
axs.set_title('Window Accuracy')
plt.show()



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
