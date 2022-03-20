import pandas as pd
import numpy as np
import scipy.io
import math

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+2)
        else:
            print('\t' * (indent+2) + str(value))

#Extracting Data Of Ex 1
ex1 = scipy.io.loadmat('S1_A1_E1.mat')
emg = ex1['emg']
EMGdf = pd.DataFrame.from_dict(emg)
stimulus = ex1['stimulus']

#Extracting Data Of Ex 2
ex2 = scipy.io.loadmat('S1_A1_E2.mat')
emg2 = ex2['emg']
EMGdf2 = pd.DataFrame.from_dict(emg2)
stimulus2 = ex2['stimulus']

#Extracting Data Of Ex 3
ex3 = scipy.io.loadmat('S1_A1_E3.mat')
emg3 = ex3['emg']
EMGdf3 = pd.DataFrame.from_dict(emg3)
stimulus3 = ex3['stimulus']


Movements = {}
#looping over the 50 movements
for m in range(1,51):

    #Checking to which exercise does this movement belong to, and retrieving its data
    if( m<11):
        movementIndices = np.where(stimulus == m)[0]
        repetitions = consecutive(movementIndices)
        EMG = EMGdf
    elif( m<28):
        movementIndices = np.where(stimulus2 == (m-10))[0]
        repetitions = consecutive(movementIndices)
        EMG = EMGdf2
    else:
        movementIndices = np.where(stimulus3 == (m-27))[0]
        repetitions = consecutive(movementIndices)
        EMG = EMGdf3


    Electrodes = {}
    #looping over the 12 electrodes of each movement
    for e in range(1,11):
        temp = {}
        for r in range(1,11):
            startIndex= repetitions[r-1][0]
            LastIndex = repetitions[r-1][len(repetitions[r-1])-1]
            df = EMG.iloc[startIndex:LastIndex, e-1]
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
    n= len(arr)
    square = 0
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
    #Calculate Mean
    mean = (square / (float)(n))
    #Calculate Root
    root = math.sqrt(mean)
    return root
def mav(arr):
    n= len(arr)
    absSum = 0
    for i in range(0,n):
        absSum += abs(arr[i])
    mav = (absSum / (float)(n))
    return mav

## Features

#Window Size = 200ms
# Overlap = 100 ms
# Step of 100 ms
#Since frequency = 100 Hz
# N= No of data points = 200ms * 100 = 20
# step size = 100ms * 100 = 10
def applyFeature(n):
    if (n==1):
        method = rms;
    elif (n==2):
        method = mav;
    Electrodes= {}
    for e in range(1,11):
        Movements={}
        for m in range(1,51):
             columnName = "Movement" + str(m)
             table = dff[columnName]['Electrode1']
             Repititions = {}
             for r in range(1,11):
                 rep = "R" + str(r)
                 windowsR = [method(table[rep][x:x+20]) for x in range(0, len(table[rep]), 10)]
                 # windowsR = pd.Series(arr)
                 Repititions["Repitition{0}".format(r)] = windowsR
             Movements["Movement{0}".format(m)]= Repititions
        Electrodes["Electrode{0}".format(e)] = Movements
    return Electrodes


Features ={ "RMS": applyFeature((1)), "MAV": applyFeature(2)}
# #Convert dictionary to dataframe
# df = pd.DataFrame.from_dict(Features)
# print(df)
# df.to_csv(r'\df.csv', index = False)
# print(Features['RMS']['Electrode1']['Movement1'])
# pretty(Features, 0)
# daily = [1,2,3,4,5,6,7,8]
# print([sum(daily[x:x+4]) for x in range(0, len(daily), 2)])


