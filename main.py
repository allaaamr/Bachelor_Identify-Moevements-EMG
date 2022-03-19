import pandas as pd
import numpy as np
import scipy.io

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

##################################################################################
# EXERCISE 1

ex1 = scipy.io.loadmat('S1_A1_E1.mat')
emg = ex1['emg']
EMGdf = pd.DataFrame.from_dict(emg)
stimulus = ex1['stimulus']

Exercise1 = {}
#looping over the 12 movements in exercise 1
for m in range(1,12):
    movementIndices = np.where(stimulus == m)[0]
    repetitions = consecutive(movementIndices)

    Electrodes = {}
    #looping over the 12 electrodes of each movement
    for e in range(1,11):
        temp = {}
        for r in range(1,11):
            startIndex= repetitions[r-1][0]
            LastIndex = repetitions[r-1][len(repetitions[r-1])-1]
            df = EMGdf.iloc[startIndex:LastIndex, e-1]
            df.reset_index(drop=True, inplace=True)
            # narray = df.to_numpy(dtype=None, copy=False)
            temp["R{0}".format(r)] = df
        Electrodes["Electrode{0}".format(e)] = temp
    Exercise1["Movement{0}".format(m)] = Electrodes

    # print(Exercise1)
    # #Convert dictionary to dataframe
    # df = pd.DataFrame.from_dict(stimulus)


##################################################################################
## EXERCISE 2

ex2 = scipy.io.loadmat('S1_A1_E2.mat')
emg2 = ex2['emg']
EMGdf2 = pd.DataFrame.from_dict(emg2)
stimulus2 = ex2['stimulus']

Exercise2 = {}
#looping over the 12 movements in exercise 1
for m in range(1,18):
    movementIndices = np.where(stimulus2 == m)[0]
    repetitions = consecutive(movementIndices)

    Electrodes = {}
    #looping over the 10 electrodes of each movement
    for e in range(1,11):
        temp = {}
        for r in range(1,11):
            startIndex= repetitions[r-1][0]
            LastIndex = repetitions[r-1][len(repetitions[r-1])-1]
            df = EMGdf2.iloc[startIndex:LastIndex, e-1]
            df.reset_index(drop=True, inplace=True)
            # narray = df.to_numpy(dtype=None, copy=False)
            temp["R{0}".format(r)] = df
        Electrodes["Electrode{0}".format(e)] = temp
    Exercise2["Movement{0}".format(m)] = Electrodes


##################################################################################
## EXERCISE 3

Exercise3 = {}
#looping over the 12 movements in exercise 1
for m in range(1,24):
    movementIndices = np.where(stimulus2 == m)[0]
    repetitions = consecutive(movementIndices)

    Electrodes = {}
    #looping over the 10 electrodes of each movement
    for e in range(1,11):
        temp = {}
        for r in range(1,11):
            startIndex= repetitions[r-1][0]
            LastIndex = repetitions[r-1][len(repetitions[r-1])-1]
            df = EMGdf2.iloc[startIndex:LastIndex, e-1]
            df.reset_index(drop=True, inplace=True)
            # narray = df.to_numpy(dtype=None, copy=False)
            temp["R{0}".format(r)] = df
        Electrodes["Electrode{0}".format(e)] = temp
    Exercise3["Movement{0}".format(m)] = Electrodes

    print(Exercise3)


## Windows of table Electrode 1
RMS = {}
for e in range(1,11):
