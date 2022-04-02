import pandas as pd
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from collections import Counter


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


Subjects_Accuracies ={}
subjects_accuracy = pd.DataFrame(columns= {'Accuracy', 'Accuracy_Modified'})
for i in range (1,11):
    subject = "S" +str(i)
    dff = pd.DataFrame.from_dict(extractSubject(subject))
    Electrodes = {}
    for e in range (1, 11):
        electrode = "Electrode" + str(e)
        M1 = dff['Movement1'][electrode]
        M11 = dff['Movement11'][electrode]
        train = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})
        test = pd.DataFrame(columns= {'RMS', 'MAV', 'VAR', 'WL', 'IAV', 'Movement'})
        for r in range(1, 7):
            rep = "R" + str(r)
            if(r in [1, 3, 4, 6]):
                df = train
            else:
                df = test

            for x in range(0, len( M1[rep]), 48):
                rms_value = rms(M1[rep][x:x + 50])
                mav_value = mav(M1[rep][x:x + 50])
                var_value = var(M1[rep][x:x + 50])
                wl_value = wl(M1[rep][x:x + 50])
                iav_value = iav(M1[rep][x:x + 50])
                movement = 0
                df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value, 'Movement':movement}
            for x in range(0, len( M11[rep]), 48):
                rms_value = rms(M11[rep][x:x + 50])
                mav_value = mav(M11[rep][x:x + 50])
                var_value = var(M11[rep][x:x + 50])
                wl_value = wl(M11[rep][x:x + 50])
                iav_value= iav(M11[rep][x:x + 50])
                movement = 1
                df.loc[df.shape[0]] = {'RMS':rms_value, 'MAV': mav_value, 'VAR':var_value, 'WL':wl_value, "IAV":iav_value, 'Movement':movement}

        lab_enc = preprocessing.LabelEncoder()
        features = {'RMS', 'MAV', 'VAR', 'WL', 'IAV'}
        X_train = pd.DataFrame.from_dict({k: train[k] for k in features})
        X_test = pd.DataFrame.from_dict({k: test[k] for k in features})
        y_train = lab_enc.fit_transform(train['Movement'])
        y_test = lab_enc.fit_transform(test['Movement'])

        clf = svm.SVC(kernel="rbf")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        y_test_new = [most_frequent(y_test[x:x + 11])  for x in range(0, len(y_test), 11)]
        y_predicted_new = [most_frequent(y_pred[x:x + 11])  for x in range(0, len(y_pred), 11)]
        accuracy_modified = accuracy_score(y_test_new, y_predicted_new)
        Electrodes["Electrode{0}".format(e)] = {'Accuracy': accuracy, 'Accuracy_Modified': accuracy_modified}
    Subjects_Accuracies["Subject{0}".format(i)] = Electrodes



pretty(Subjects_Accuracies)
electrode1_acc = []; electrode1_mod = []; electrode2_acc = []; electrode2_mod = []; electrode3_acc = []; electrode3_mod = []; electrode4_acc = [];
electrode4_mod = []; electrode5_acc = []; electrode5_mod = []; electrode6_acc = []; electrode6_mod = []; electrode7_acc = []; electrode7_mod = [];
electrode8_acc = []; electrode8_mod = []; electrode9_acc = []; electrode9_mod = []; electrode10_acc = []; electrode10_mod = []

for d in Subjects_Accuracies.values():
    electrode1_acc.append(round(d['Electrode1']['Accuracy'],2))
    electrode1_mod.append(d['Electrode1']['Accuracy_Modified'])

    electrode2_acc.append(round(d['Electrode2']['Accuracy'],2))
    electrode2_mod.append(d['Electrode2']['Accuracy_Modified'])

    electrode3_acc.append(round(d['Electrode3']['Accuracy'],2))
    electrode3_mod.append(d['Electrode3']['Accuracy_Modified'])

    electrode4_acc.append(round(d['Electrode4']['Accuracy'],2))
    electrode4_mod.append(d['Electrode4']['Accuracy_Modified'])

    electrode5_acc.append(d['Electrode5']['Accuracy'])
    electrode5_mod.append(d['Electrode5']['Accuracy_Modified'])

    electrode6_acc.append(d['Electrode6']['Accuracy'])
    electrode6_mod.append(d['Electrode6']['Accuracy_Modified'])

    electrode7_acc.append(d['Electrode7']['Accuracy'])
    electrode7_mod.append(d['Electrode7']['Accuracy_Modified'])

    electrode8_acc.append(d['Electrode8']['Accuracy'])
    electrode8_mod.append(d['Electrode8']['Accuracy_Modified'])

    electrode9_acc.append(d['Electrode9']['Accuracy'])
    electrode9_mod.append(d['Electrode9']['Accuracy_Modified'])

    electrode10_acc.append(d['Electrode10']['Accuracy'])
    electrode10_mod.append(d['Electrode10']['Accuracy_Modified'])


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


subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

x = np.arange(len(subjects))  # the label locations
width = 0.08  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 9*width/2, electrode1_mod, width, label='Electrode 1')
rects2 = ax.bar(x - 7*width/2, electrode2_mod, width, label='Electrode 2')
rects3 = ax.bar(x - 5*width/2, electrode3_mod, width, label='Electrode 3')
rects4 = ax.bar(x - 3*width/2, electrode4_mod, width, label='Electrode 4')
rects5 = ax.bar(x - width/2, electrode5_mod, width, label='Electrode 5')
rects6 = ax.bar(x + width/2, electrode6_mod, width, label='Electrode 6')
rects7 = ax.bar(x + 3*width/2, electrode7_mod, width, label='Electrode 7')
rects8 = ax.bar(x + 5*width/2, electrode8_mod, width, label='Electrode 8')
rects9 = ax.bar(x + 7*width/2, electrode9_mod, width, label='Electrode 9')
rects10 = ax.bar(x + 9*width/2, electrode10_mod, width, label='Electrode 10')

ax.set_ylabel('Accuracy')
ax.set_title('Movement Accuracy')
ax.set_xticks(x, subjects)
ax.legend()
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)
fig.tight_layout()
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
