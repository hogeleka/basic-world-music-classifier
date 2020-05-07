import librosa
import librosa.display
import IPython.display as ipd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
import time

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras import models
from keras import layers
#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

sampleRate = 88
desiredDuration = 30

def create1DArrayFromSpecData(fileName, numpy2dArray, classification):
    output = []
    output.append(fileName)
    for row in numpy2dArray:
        for value in row:
            output.append(value)
    output.append(classification)
    return output


def readGenres():
    filename = f'data_{sampleRate}.csv'
    file = open(filename, 'w', newline='\n')
    correctSize = (sampleRate / 2) * desiredDuration
    with file:
        writer = csv.writer(file)
        genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        for g in genres:
            for filename in os.listdir(f'genres/{g}'):
                songname = f'genres/{g}/{filename}'
                # if songname.endswith('9.wav'):
                y, sr = librosa.load(songname, sr=(sampleRate//2), duration=desiredDuration) #//librosa quirks
                yConverted = librosa.stft(y)
                ampToDB = librosa.amplitude_to_db(abs(yConverted))
                csvRow = create1DArrayFromSpecData(songname, ampToDB, g)
                writer.writerow(csvRow)
                # print(f'finished loading data for {songname}')
                # if y.size != correctSize:
                #     print(f'{songname} is fucking useless. it has {y.size} data points instead of {correctSize}')
            print(f'-------------finished loading {g} songs------------')
                # assert y.size()
                # print(y, sr)

# def readCSVFiles():
#     data = pd.read_csv('data.csv', header=None)
#     data.head()
# audio_path = "genres/blues/blues.00000.wav"
# x , sr = librosa.load(audio_path)
# # print(x.shape, sr)
# librosa.load(audio_path, sr=44100)
# # plt.figure(figsize=(14, 5))
# # librosa.display.waveplot(x, sr=sr)
# # ipd.Audio(audio_path)
#
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# # plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# # plt.colorbar()
# # plt.show()
#
#
# audio_path = "genres/classical/classical.00000.wav"
# x1 , sr = librosa.load(audio_path, sr=44100)
# # print(x.shape, sr)
# # librosa.load(audio_path, sr=44100)
# # plt.figure(figsize=(14, 5))
# # librosa.display.waveplot(x, sr=sr)
# # ipd.Audio(audio_path)
#
# X1 = librosa.stft(x1)
# X1db = librosa.amplitude_to_db(abs(X1))
# # plt.figure(figsize=(14, 5))
# # librosa.display.specshow(X1db, sr=sr, x_axis='time', y_axis='hz')
# # plt.colorbar()
# # plt.show()
# X1dbConverted = create1DArrayFromSpecData("test", X1db, "test")
# assert len(X1dbConverted) == 1325325 + 1

start = time.time()
readGenres()
print(f'----finished reading files and writing csv in {time.time() - start} seconds---')

start2 = time.time()
data = pd.read_csv(f'data_{sampleRate}.csv', header=None, usecols=[i for i in range(1, sampleRate+1)])
data.head()
print(data.shape)

# Encode labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# Scaling feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

# Divide into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'{len(y_train)} rows for training and {len(y_test)} rows for testing')

print(f'----finished divinding csv data to train and test in {time.time() - start2} seconds---')


# Building network / Keras Classification
start3 = time.time()
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

print(f'----finished doing the epochs in {time.time() - start3} seconds---')


test_loss, test_acc = model.evaluate(X_test,y_test)
print(f'test accuracy: {test_acc}')



# songname = "genres/blues/blues.00004.wav"
# y, sr = librosa.load(songname, sr=(sampleRate//2), duration=desiredDuration)
