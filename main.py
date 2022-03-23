import pandas as pd
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt


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
# looping over the 50 movements
for m in range(1, 51):

    # Checking to which exercise does this movement belong to, and retrieving its data
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

#Convert dictionary to dataframe
dff = pd.DataFrame.from_dict(Movements)
# print(dff)
# print(dff.loc['Electrode1']['Movement1'])

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4)
fig.suptitle('Electrode 1 Movement 2')

ax1.set_title('R1')
ax1.plot(dff.loc['Electrode1']['Movement2']['R1'])

ax2.set_title('R2')
ax2.plot(dff.loc['Electrode1']['Movement2']['R2'])

ax3.set_title('R3')
ax3.plot(dff.loc['Electrode1']['Movement2']['R3'])

ax4.set_title('R4')
ax4.plot(dff.loc['Electrode1']['Movement2']['R4'])

ax5.set_title('R5')
ax5.plot(dff.loc['Electrode1']['Movement2']['R5'])

ax6.set_title('R6')
ax6.plot(dff.loc['Electrode1']['Movement2']['R6'])

ax7.set_title('R7')
ax7.plot(dff.loc['Electrode1']['Movement2']['R7'])

ax8.set_title('R8')
ax8.plot(dff.loc['Electrode1']['Movement2']['R8'])


plt.show()

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


#Extracting Features
electrodes = {}
for e in range(1, 11):
    movements = {}
    for m in range(1, 51):
        columnName = "Movement" + str(m)
        table = dff[columnName]['Electrode1']
        Features={}
        RMSrep = []
        MAVrep = []
        VARrep = []
        WLrep =[]
        IAVrep = []
        for r in range(1, 11):
            rep = "R" + str(r)
            rmsArr = [rms(table[rep][x:x + 50]) for x in range(0, len(table[rep]), 48)]
            mavArr = [mav(table[rep][x:x + 50]) for x in range(0, len(table[rep]), 48)]
            varArr = [var(table[rep][x:x + 50]) for x in range(0, len(table[rep]), 48)]
            wlArr = [wl(table[rep][x:x + 50]) for x in range(0, len(table[rep]), 48)]
            iavArr = [iav(table[rep][x:x + 50]) for x in range(0, len(table[rep]), 48)]
        #    windowsR = pd.Series(arr)

            RMSrep = np.concatenate((RMSrep,rmsArr ), axis=None)
            MAVrep = np.concatenate((MAVrep,mavArr ), axis=None)
            VARrep = np.concatenate((VARrep,varArr ), axis=None)
            WLrep = np.concatenate((WLrep,wlArr ), axis=None)
            IAVrep = np.concatenate((IAVrep,iavArr ), axis=None)

        Features['RMS'] = RMSrep
        Features['MAV'] = MAVrep
        Features['VAR'] = VARrep
        Features['WL'] = WLrep
        Features['IAV'] = IAVrep

        movements["Movement{0}".format(m)] = Features
    electrodes["Electrode{0}".format(e)] = movements



# Features = {"RMS": applyFeature((1)), "MAV": applyFeature(2)}
# #Convert dictionary to dataframe

# df = pd.DataFrame.from_dict(electrodes)
# E1M1 = pd.DataFrame.from_dict(df['Electrode1']['Movement1'])
# E1M2 = pd.DataFrame.from_dict(df['Electrode1']['Movement2'])
# E1M3 = pd.DataFrame.from_dict(df['Electrode1']['Movement3'])
# E1M4 = pd.DataFrame.from_dict(df['Electrode1']['Movement4'])
# E1M5 = pd.DataFrame.from_dict(df['Electrode1']['Movement5'])
# E1M6 = pd.DataFrame.from_dict(df['Electrode1']['Movement6'])
#
# print(E1M1)
# stats_df = E1M1.describe()
# stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
# out_fields = ['mean','25%','50%','75%', 'range']
# stats_df = stats_df.loc[out_fields]
# stats_df.rename({'50%': 'median'}, inplace=True)
# print(stats_df)

# ax = plt.axes()
#
# ax.scatter(E1M1.IAV, E1M1.MAV)
#
# # Label the axes
# ax.set(xlabel='IAV  (cm)',
#        ylabel='MAV  (cm)',
#        title='IAV vs MAV');
#
# plt.show()

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# fig.suptitle('RMS')
#
# ax1.set_title('Electrode 1 Movement 1')
# ax1.plot(E1M1['RMS'])
#
# ax2.set_title('Electrode 1 Movement 2 ')
# ax2.plot(E1M2['RMS'])
#
# ax3.set_title('Electrode 1 Movement 3 ')
# ax3.plot(E1M3['RMS'])
#
# ax4.set_title('Electrode 1 Movement 4 ')
# ax4.plot(E1M4['RMS'])
#
# ax5.set_title('Electrode 1 Movement 5 ')
# ax5.plot(E1M5['RMS'])
#
# ax6.set_title('Electrode 1 Movement 6 ')
# ax6.plot(E1M6['RMS'])
#
# plt.show()
#ax.scatter(data.sepal_length, data.sepal_width)
#dfRMS.to_csv(r'\df.csv', index = False)
#pretty(electrodes)
#pretty(df['Electrode4']['Movement9'])
# df = pd.DataFrame.from_dict(Features)
# df = pd.DataFrame.from_dict(Features)
# print(df['RMS'])
# df.to_csv(r'\df.csv', index = False)
# print(Features['RMS']['Electrode1']['Movement1'])
# pretty(Features, 0)
