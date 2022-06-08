import pandas as pd
import numpy as np
import scipy.io
import math
from collections import Counter
import copy
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Conv2D, Flatten, Dropout, MaxPooling2D,Conv1D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import BatchNormalization
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
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
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
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
    for m in range(1, 53):
        if (m < 13):
            movementIndices = np.where(stimulus == m)[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf
        elif (m < 30):
            movementIndices = np.where(stimulus2 == (m - 12))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf2
        else:
            movementIndices = np.where(stimulus3 == (m - 29))[0]
            repetitions = consecutive(movementIndices)
            EMG = EMGdf3

        Repetitions = {}
        for r in range(1, 7):
            startIndex = repetitions[r - 1][0]
            LastIndex = repetitions[r - 1][len(repetitions[r - 1]) - 1]
            df = EMG.iloc[startIndex:LastIndex, 0:10]
            df.reset_index(drop=True, inplace=True)
            Repetitions["R{0}".format(r)] = df
        Movements["Movement{0}".format(m)] = Repetitions
    return Movements
def Average(lst):
    return sum(lst) / len(lst)



columns = {'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'Train','Movement'}
X_train = []
y_train = []
X_test = []
y_test = []
for s in range(1,28):
    subject = 'S' + str(s)
    dff = pd.DataFrame.from_dict(extractSubject(subject))
    for m in range(1,11):
    
        M = dff['Movement'+str(m)]
        for r in range(1,7): 
            r_index = "R" + str(r)
            rep = M[r_index]
            if (r in [1, 3, 4, 6]):
                train = True
            else:
                train = False
            i=0
            for x in range(0, len(rep), 6):
                entry = rep.iloc[x:x+15, :]
                entry.columns = {'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'}
                entry.reset_index(drop=True, inplace=True)
                matrix = entry.to_numpy()
                if matrix.shape[0]==15:
                    if(train):
                        X_train.append(matrix)
                        y_train.append(m)
                    else :
                        X_test.append(matrix)
                        y_test.append(m)  
                    i+=1
                    print(i)    

# X_train = np.array(X_train)
# X_test = np.array(X_test)
# print("X Train shape: ", X_train.shape)
# print("X Test shape: ", X_test.shape)
# print('------------------')
# X_train = np.array(X_train).reshape(X_train.shape[0],15,10)
# X_test = np.array(X_test).reshape(X_test.shape[0],15,10)
# y_train = np.array(y_train).astype('float32').reshape(X_train.shape[0],1)
# y_train= tf.one_hot(y_train,52)
# y_test = np.array(y_test).astype('float32').reshape(X_test.shape[0],1)
# y_test= tf.one_hot(y_test,52)
# y_test = np.array(y_test).astype('float32').reshape(X_test.shape[0],52)
# y_train = np.array(y_train).astype('float32').reshape(X_train.shape[0],52)

# print("X Train shape: ", X_train.shape)
# print("X Test shape: ", X_test.shape)
# print("Y Train shape: ", y_train.shape)
# input = Input(shape =(100,15,10,1))
# x = Conv2D(32,(1,10), kernel_initializer='glorot_normal', activation='relu', padding='same', input_shape = (15,10))(input)
# x = Dropout(0.15)(x)

# x = Conv2D(32,(3,3),kernel_initializer='glorot_normal', activation='relu' ,padding='same')(x)
# x = Dropout(0.15)(x)
# x = MaxPooling2D((3,3))(x)

# x = Conv2D(64,(5,5), kernel_initializer='glorot_normal', activation='relu', padding='same')(x)
# x = Dropout(0.15)(x)
# x = MaxPooling2D((3,3))(x)

# x = Conv2D(64,(5,1), kernel_initializer='glorot_normal', activation='relu', padding='same')(x)
# x = Dropout(0.15)(x)
# # x = Dense(64,1, activation='softmax', padding='valid')(x)
# x = Flatten()(x)
# ouput = Dense(52, kernel_initializer='glorot_normal', activation='softmax')(x)


# model = Model(input, ouput)
# model.compile(optimizer=SGD(learning_rate=0.05), loss="categorical_crossentropy", metrics=['accuracy'])
# print(X_train.shape)

# model.fit(X_train, y_train, epochs=100, validation_data= (X_test, y_test))
# print(model.evaluate(X_test, y_test))

