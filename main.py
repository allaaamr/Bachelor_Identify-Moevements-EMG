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

# Extracting Data Of Ex 1
ex1 = scipy.io.loadmat('S1_A1_E1.mat')
emg = ex1['emg']
EMGdf = pd.DataFrame.from_dict(emg)
stimulus = ex1['stimulus']

# Extracting Data Of Ex 2
ex2 = scipy.io.loadmat('S1_A1_E2.mat')
emg2 = ex2['emg']
EMGdf2 = pd.DataFrame.from_dict(emg2)
stimulus2 = ex2['stimulus']

# Extracting Data Of Ex 3
ex3 = scipy.io.loadmat('S1_A1_E3.mat')
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

dff = pd.DataFrame.from_dict(Movements)
#Extract Repetitions of Electrode 1:  Ex1M1
M1Ex1 = dff['Movement1']['Electrode1']
M1Ex2 = dff['Movement11']['Electrode1']

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

# 1 repetition = 11 windows
# 6 repetitions = 66 windows
# Each exercise has 66 rows in the dataset

train = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})
test = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})


for r in range(1, 7):
    rep = "R" + str(r)
    if(r in [1, 3, 4, 6]):
        df = train
    else:
        df = test

    for x in range(0, len( M1Ex1[rep]), 48):
        rms_value = rms(M1Ex1[rep][x:x + 50])
        mav_value = mav(M1Ex1[rep][x:x + 50])
        var_value = var(M1Ex1[rep][x:x + 50])
        wl_value = wl(M1Ex1[rep][x:x + 50])
        iav_value= iav(M1Ex1[rep][x:x + 50])
        movement = 0
        df.loc[df.shape[0]] = [rms_value, mav_value, var_value, wl_value, iav_value, movement]

    for x in range(0, len( M1Ex2[rep]), 48):
        rms_value = rms(M1Ex2[rep][x:x + 50])
        mav_value = mav(M1Ex2[rep][x:x + 50])
        var_value = var(M1Ex2[rep][x:x + 50])
        wl_value = wl(M1Ex2[rep][x:x + 50])
        iav_value= iav(M1Ex2[rep][x:x + 50])
        movement = 1
        df.loc[df.shape[0]] = [rms_value, mav_value, var_value, wl_value, iav_value, movement]

lab_enc = preprocessing.LabelEncoder()
features = {'RMS', 'MAV', 'VAR', 'WL', 'IAV'}
X_train = pd.DataFrame.from_dict({k: train[k] for k in features})
X_test = pd.DataFrame.from_dict({k: test[k] for k in features})
y_train = lab_enc.fit_transform(train['Movement'])
y_test = lab_enc.fit_transform(test['Movement'])
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=0)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# df = pd.DataFrame.from_dict(electrodes)
# E1M1 = pd.DataFrame.from_dict(df['Electrode1']['Movement1'])
# # E1M2 = pd.DataFrame.from_dict(df['Electrode1']['Movement2'])
# # E1M3 = pd.DataFrame.from_dict(df['Electrode1']['Movement3'])
# # E1M4 = pd.DataFrame.from_dict(df['Electrode1']['Movement4'])
# # E1M5 = pd.DataFrame.from_dict(df['Electrode1']['Movement5'])
# # E1M6 = pd.DataFrame.from_dict(df['Electrode1']['Movement6'])
#
# print(E1M1)
# print("----------------------------------------------")
# stats_df = E1M1.describe()
# stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# out_fields = ['mean','25%','50%','75%', 'range']
# stats_df = stats_df.loc[out_fields]
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
