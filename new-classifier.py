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
from keras.models import load_model

from keras import models
from keras import layers
#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

# sampleRate = 88
# desiredDuration = 30

# def predictIndividualFile(songName, model):
#     y, sr = librosa.load(songName, mono=True, duration=30)
#     dataRow = []
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     dataRow.append(np.mean(chroma_stft))
#     rmse = librosa.feature.rms(y=y)
#     dataRow.append(np.mean(rmse))
#     spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     dataRow.append(np.mean(spec_cent))
#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     dataRow.append(np.mean(spec_bw))
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     dataRow.append(np.mean(rolloff))
#     zcr = librosa.feature.zero_crossing_rate(y)
#     dataRow.append(np.mean(zcr))
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     for e in mfcc:
#         dataRow.append(np.mean(e))
#     dataRowNumpyArray = np.array(dataRow)
#     # transformedData = []
#     # for val in dataRow:
#     #     valRow = [val]
#     #     transformedData.append(valRow)
#     # print(len(dataRowNumpyArray))
#     # print(f'transformedData: ', transformedData)
#     # frame = pd.DataFrame()
#     # dataRowNumpyArray.iloc[:, :-1], dtype=float)
#     # dataRowWrapped = np.array(dataRowNumpyArray)
#     # dataRowNumpyArray = np.array(transformedData)
#     predicted = model.predict(np.array([dataRowNumpyArray, ]))
#     # predicted = model.predict_classes(dataRowNumpyArray)
#
#     print(f'predicted class: {np.argmax(predicted[0])}')
#     return predicted


csvDataFileName = "outputData_new.csv"

#prepare csv file header
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()



#extracting every audio

# write header row to file
start0 = time.time()
file = open(csvDataFileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
#
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# def extractAudioSpecs():
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'genres/{g}'):
        songname = f'genres/{g}/{filename}'
        trackNumber = int(songname[-6:-4])
        print(f'---reading {g} track {trackNumber}------')
        if trackNumber < 90:
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open(csvDataFileName, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        else:
            break
    print(f'----done with {g}-------')

end0 = time.time()
print(f'----finished writing csv in {end0 - start0} seconds------')

start1 = time.time()
data = pd.read_csv(csvDataFileName)
data.head()

print(data.shape)

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)


#Encoding the labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaling Feature Columns
scaler = StandardScaler()
X = np.array(data.iloc[:, :-1], dtype=float)
# X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

#Divide into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
end1 = time.time()
print(f'------finished reading, processing, and dividing up data in {end1-start1} seconds--------')

start2 = time.time()

#building network
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
                    epochs=10000,
                    batch_size=128)

test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)
end2 = time.time()
print(f'------------finished with epochs and stuff in {end2-start2}seconds -------------')
model.summary()
modelFileName = "musicGenreModel_1000New.h5"
modelSaved = model.save(modelFileName)
# modelSaved.summary()

# testFile1 = "sample_audio/Dead Flower Complete.wav"
# testFile2 = "sample_audio/06 Swasti Path 2 Final.mp3"
# predictIndividualFile(testFile1, model)
# predictIndividualFile(testFile2, model)

# count = 0
# for file in os.listdir("genres/metal"):
#     songName = f"genres/metal/{file}"
#     print(f"predicting {songName}-------")
#     predictIndividualFile(songName, model)
#     print(f"finished predicting {songName} --------------------")
#     count += 1
#     if count == 20:
#         break
# bluesSongs =
