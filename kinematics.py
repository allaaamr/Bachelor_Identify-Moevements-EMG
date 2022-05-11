import pandas as pd
import numpy as np
import scipy.io
import math

def rms(arr):
    n = len(arr)
    squared = np.array(arr) * np.array(arr)
    sum = np.sum(squared)
    mean = (sum / (float)(n))
    root = math.sqrt(mean)
    return root
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):

            pretty(value, indent + 2)
            print('-----------------------')
        else:
            print('\t' * (indent + 2) + str(value))
            print(" ")
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
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
def extractSubjectAngles(name):
    ex1Path = 'Kinematics/' + name + '/' + name + '_E1_A1.mat'
    print(ex1Path)
    ex1 = scipy.io.loadmat(ex1Path)
    angles1 = ex1['angles']
    anglesDF1 = pd.DataFrame.from_dict(angles1)
    stimulus = ex1['restimulus']

    ex2Path = 'Kinematics/' + name + '/' + name + '_E2_A1.mat'
    ex2 = scipy.io.loadmat(ex2Path)
    angles2 = ex2['angles']
    anglesDF2 = pd.DataFrame.from_dict(angles2)
    stimulus2 = ex2['restimulus']

    ex3Path = 'Kinematics/' + name + '/' + name + '_E3_A1.mat'
    ex3 = scipy.io.loadmat(ex3Path)
    angles3 = ex3['angles']
    anglesDF3 = pd.DataFrame.from_dict(angles3)
    stimulus3 = ex3['restimulus']

    CMC1_A = {}
    for m in range(1, 51):
        if (m <= 12):
            movementIndices = np.where(stimulus == m)[0]
            repetitions = consecutive(movementIndices)
            angle = anglesDF1
        elif (m <= 29):
            movementIndices = np.where(stimulus2 == (m - 12))[0]
            repetitions = consecutive(movementIndices)
            angle = anglesDF2
        else:
            movementIndices = np.where(stimulus3 == (m - 29))[0]
            repetitions = consecutive(movementIndices)
            angle = anglesDF3

        temp = {}
        for r in range(1, 7):
            startIndex = repetitions[r - 1][0]
            LastIndex = repetitions[r - 1][len(repetitions[r - 1]) - 1]
            df = angle.iloc[startIndex:LastIndex, 1]
            df.reset_index(drop=True, inplace=True)
            narray = df.to_numpy(dtype=None, copy=False)
            temp["R{0}".format(r)] = narray

        CMC1_A["Movement{0}".format(m)] = temp
    return CMC1_A

print(pretty(extractSubject('S1')))

final_df = pd.DataFrame(columns={'RMS1', 'RMS2', 'RMS3', 'RMS4', 'RMS5','RMS6', 'RMS7',
                                 'RMS8', 'RMS9', 'RMS10','Train','Movement','CMC1_A'})

subject = 'S' + str(1)
dff = pd.DataFrame.from_dict(extractSubject(subject))
df = pd.DataFrame(columns={'RMS1', 'RMS2', 'RMS3', 'RMS4', 'RMS5','RMS6', 'RMS7',
                           'RMS8', 'RMS9', 'RMS10','Train','Movement','CMC1_A'})

df_angles = pd.DataFrame.from_dict(extractSubjectAngles(subject))
for e in range(1, 11):
    i = 0
    electrode = 'Electrode' + str(e)

    for m in range(1,51):
        M = dff['Movement'+str(m)][electrode]
        Angles = df_angles['Movement'+str(m)]
        for r in range(1, 7):
            rep = "R" + str(r)
            if (r in [1, 3, 4, 6]):
                train = 1
            else:
                train = 0

            print("Start" + str(i))
            print("LLL" + str(len(M[rep])))
            for x in range(0, len(M[rep]), 48):
                df.at[i, 'RMS' + str(e)] = rms(M[rep][x:x + 50])
                df.at[i, 'Movement'] = m
                df.at[i, 'Train'] = train
                i += 1

            print("End" + str(i))
            i -= 11
            print("Restart" + str(i))
            print("LLL" + str(len(Angles[rep])))
            for x in range(0, len(Angles[rep]), 48):
                df.at[i, 'CMC1_A'] = rms(Angles[rep][x:x + 50])
                i += 1
            print("Re-end" + str(i))

final_df = final_df.append(df, ignore_index=True)
print(final_df)