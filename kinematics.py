import pandas as pd
import numpy as np
import scipy.io
import math
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

warnings.filterwarnings("ignore")


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
def rms(arr):
    n = len(arr)
    squared = np.array(arr) * np.array(arr)
    sum = np.sum(squared)
    mean = (sum / (float)(n))
    root = math.sqrt(mean)
    return root
def mav(arr):
    n = len(arr)
    mav = sum((abs(np.array(arr))))/(float)(n)
    return mav
def var(arr):
    n = len(arr)
    squared = np.array(arr) * np.array(arr)
    sum = np.sum(squared)
    result = (sum / (float)(n))
    return result
def wl(arr):
    n = len(arr)
    sum = 0
    for i in range(1, n):
        sum += abs(arr[i] - arr[i - 1])
    return sum
def iav(arr):
    return sum(abs(np.array(arr)))
def mean(arr):
    return np.sum(arr)/len(arr)
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
    stimulus = ex1['restimulus']

    ex2Path = 'DB1/' + name + '/' + name + '_A1_E2.mat'
    ex2 = scipy.io.loadmat(ex2Path)
    emg2 = ex2['emg']
    EMGdf2 = pd.DataFrame.from_dict(emg2)
    stimulus2 = ex2['restimulus']

    ex3Path = 'DB1/' + name + '/' + name + '_A1_E3.mat'
    ex3 = scipy.io.loadmat(ex3Path)
    emg3 = ex3['emg']
    EMGdf3 = pd.DataFrame.from_dict(emg3)
    stimulus3 = ex3['restimulus']

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
            for r in range(1, 7):
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
        if (m < 11):
            movementIndices = np.where(stimulus == m)[0]
            repetitions = consecutive(movementIndices)
            angle = anglesDF1
        elif (m < 28):
            movementIndices = np.where(stimulus2 == (m - 10))[0]
            repetitions = consecutive(movementIndices)
            angle = anglesDF2
        else:
            movementIndices = np.where(stimulus3 == (m - 27))[0]
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

    # CMC1_A = {}
    # for m in range(1, 51):
    #     if (m < 11):
    #         movementIndices = np.where(stimulus == m)[0]
    #         repetitions = consecutive(movementIndices)
    #         angle = anglesDF1
    #     elif (m < 28):
    #         movementIndices = np.where(stimulus2 == (m - 10))[0]
    #         repetitions = consecutive(movementIndices)
    #         angle = anglesDF2
    #     else:
    #         movementIndices = np.where(stimulus3 == (m - 27))[0]
    #         repetitions = consecutive(movementIndices)
    #         angle = anglesDF3
    #
    #     temp = {}
    #     for r in range(1, 7):
    #         startIndex = repetitions[r - 1][0]
    #         LastIndex = repetitions[r - 1][len(repetitions[r - 1]) - 1]
    #         df = angle.iloc[startIndex:LastIndex, 1]
    #         df.reset_index(drop=True, inplace=True)
    #         narray = df.to_numpy(dtype=None, copy=False)
    #         temp["R{0}".format(r)] = narray
    #
    #     CMC1_A["Movement{0}".format(m)] = temp
    return CMC1_A

final_df = pd.DataFrame(columns={'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
                                 'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
                                 'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
                                 'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
                                 'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
                                 'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
                                 'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
                                 'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
                                 'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
                                 'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10',
                                 'Train'})

df_angle = pd.DataFrame(columns={'CMC1_A'})

i = 0
for s in range(1,8):
    subject = 'S' + str(s)
    df_angles = pd.DataFrame.from_dict(extractSubjectAngles(subject))
    for m in range(1,51):
        M = df_angles['Movement'+str(m)]
        for r in range(1,7):
            rep = "R" + str(r)
            for x in range(0, len(M[rep]), 48):
                df_angle.at[i, 'CMC1_A'] = mean(M[rep][x:x + 50])
                i += 1

for s in range(1,8):
    subject = 'S' + str(s)
    dff = pd.DataFrame.from_dict(extractSubject(subject))
    df = pd.DataFrame(columns={'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
                           'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
                           'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
                           'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
                           'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
                           'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
                           'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
                           'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
                           'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
                           'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10',
                           'Train'})
    for e in range(1, 11):
        electrode = 'Electrode' + str(e)
        i=0
        for m in range(1,51):
            M = dff['Movement'+str(m)][electrode]
            for r in range(1, 7):
                rep = "R" + str(r)
                if (r in [1, 3, 4, 6]):
                    train = 1
                else:
                    train = 0
                for x in range(0, len(M[rep]), 48):
                    df.at[i, 'RMS' + str(e)] = rms(M[rep][x:x + 50])
                    df.at[i, 'MAV' + str(e)] = mav(M[rep][x:x + 50])
                    df.at[i, 'VAR' + str(e)] = var(M[rep][x:x + 50])
                    df.at[i, 'WL' + str(e)] = wl(M[rep][x:x + 50])
                    df.at[i, 'IAV' + str(e)] = iav(M[rep][x:x + 50])
                    df.at[i, 'Train'] = train
                    i += 1
    final_df = final_df.append(df, ignore_index=True)

final_df['CMC1_A'] = df_angle['CMC1_A']
print(df)


features = {'RMS1', 'MAV1', 'VAR1', 'WL1', 'IAV1',
            'RMS2', 'MAV2', 'VAR2', 'WL2', 'IAV2',
            'RMS3', 'MAV3', 'VAR3', 'WL3', 'IAV3',
            'RMS4', 'MAV4', 'VAR4', 'WL4', 'IAV4',
            'RMS5', 'MAV5', 'VAR5', 'WL5', 'IAV5',
            'RMS6', 'MAV6', 'VAR6', 'WL6', 'IAV6',
            'RMS7', 'MAV7', 'VAR7', 'WL7', 'IAV7',
            'RMS8', 'MAV8', 'VAR8', 'WL8', 'IAV8',
            'RMS9', 'MAV9', 'VAR9', 'WL9', 'IAV9',
            'RMS10', 'MAV10', 'VAR10', 'WL10', 'IAV10'}

X_train = df[df['Train'] == 1].loc[:, features]
X_test = df[df['Train'] == 0].loc[:, features]
y_train = df[df['Train'] == 1]['CMC1_A'].astype('int')
y_test = df[df['Train'] == 0]['CMC1_A'].astype('int')
print(X_train.shape)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
y_predicted_new = [most_frequent(y_pred[x:x + 11]) for x in range(0, len(y_pred), 11)]
accuracy_modified = accuracy_score(y_test_new, y_predicted_new)

# input = Input()



# x = final_df.loc[:, features]
# y = final_df.loc[:,['CMC1_A']].values
# y=y.astype('int')
# x = StandardScaler().fit_transform(x)

# for p in range(5,51):
#     pca = PCA(n_components=p)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(data=principalComponents)
#     finalDf = pd.concat([principalDf, final_df['CMC1_A'], final_df['Train']], axis=1)

#     X_train = finalDf[finalDf['Train'] == 1]
#     X_train.drop({'CMC1_A', 'Train'}, axis=1, inplace=True)
#     X_test = finalDf[finalDf['Train'] == 0]
#     X_test.drop({'CMC1_A', 'Train'}, axis=1, inplace=True)
#     y_train = finalDf[finalDf['Train'] == 1]['CMC1_A'].astype('int')
#     y_test = finalDf[finalDf['Train'] == 0]['CMC1_A'].astype('int')

#     clf = KNeighborsClassifier(n_neighbors=1)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     y_test_new = [most_frequent(y_test[x:x + 11]) for x in range(0, len(y_test), 11)]
#     y_predicted_new = [most_frequent(y_pred[x:x + 11]) for x in range(0, len(y_pred), 11)]
#     accuracy_modified = accuracy_score(y_test_new, y_predicted_new)

#     print("Window Accuracy",accuracy)
#     print("Movement Accuracy", accuracy_modified)