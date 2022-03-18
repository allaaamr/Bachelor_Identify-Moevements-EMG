import pandas as pd
import numpy as np
import scipy.io

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

data = scipy.io.loadmat('S1_A1_E1.mat')
emg = data['emg']
EMGdf = pd.DataFrame.from_dict(emg)
stimulus = data['stimulus']

Exercise1 = {}
#looping over the 12 movements in exercise 1
for m in range(1,12):
    movementIndices = np.where(stimulus == m)[0]
    repetitions = consecutive(movementIndices)

    Electrodes = {}
    #looping over the 12 electrodes of each movement
    for e in range(1,11):
        temp = {}
        for x in range(0,10):
            startIndex= repetitions[x][0]
            LastIndex = repetitions[x][len(repetitions[x])-1]
            df = EMGdf.iloc[startIndex:LastIndex, e-1]
            df.reset_index(drop=True, inplace=True)
            # narray = df.to_numpy(dtype=None, copy=False)
            temp["R{0}".format(x)] = df
        Electrodes["Electrode{0}".format(e)] = temp
    Exercise1["Movement{0}".format(m)] = Electrodes

    print(Exercise1)
    #Convert dictionary to dataframe
    df = pd.DataFrame.from_dict(stimulus)


