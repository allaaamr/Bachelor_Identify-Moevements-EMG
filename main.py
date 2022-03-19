import pandas as pd
import numpy as np
import scipy.io

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

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
            temp["R{0}".format(r)] = df
        Electrodes["Electrode{0}".format(e)] = temp
    Movements["Movement{0}".format(m)] = Electrodes


# #Convert dictionary to dataframe
dff = pd.DataFrame.from_dict(Movements)
print(dff)
# print(dff['Movement1'])
# print(dff.loc['Electrode1'])






# ## Windows of table Electrode 1
# RMS = {}
# for e in range(1,11):
