import pandas as pd
import numpy as np
import scipy.io
import math


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

# #Convert dictionary to dataframe
dff = pd.DataFrame.from_dict(Movements)


# print(dff)
# print(dff.loc['Electrode1'])


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


## Features
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
        for r in range(1, 11):
            rep = "R" + str(r)
            rmsArr = [rms(table[rep][x:x + 600]) for x in range(0, len(table[rep]), 400)]
            mavArr = [mav(table[rep][x:x + 600]) for x in range(0, len(table[rep]), 400)]
            varArr = [var(table[rep][x:x + 600]) for x in range(0, len(table[rep]), 400)]
        #    windowsR = pd.Series(arr)

            RMSrep.append(rmsArr)
            MAVrep.append(mavArr)
            VARrep.append(varArr)
        Features['RMS'] = RMSrep
        Features['MAV'] = MAVrep
        Features['VAR'] = VARrep
        movements["Movement{0}".format(m)] = Features
    electrodes["Electrode{0}".format(e)] = movements



# Features = {"RMS": applyFeature((1)), "MAV": applyFeature(2)}
# #Convert dictionary to dataframe

df = pd.DataFrame.from_dict(electrodes)
#dfRMS.to_csv(r'\df.csv', index = False)
pretty(electrodes)
#pretty(df['Electrode4'])
# df = pd.DataFrame.from_dict(Features)
# df = pd.DataFrame.from_dict(Features)
# print(df['RMS'])
# df.to_csv(r'\df.csv', index = False)
# print(Features['RMS']['Electrode1']['Movement1'])
# pretty(Features, 0)
