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

modelFileName = "musicGenreModel_1000New.h5"
# genre = "rock"
# testFile1 = "sample_audio/Dead Flower Complete.wav"
# testFile2 = "genres/country/country.00099.wav"
# testFile3 = f'genres/{genre}/{genre}.00008.wav'
# testFile2 = "sample_audio/06 Swasti Path 2 Final.mp3"
# loadedModel = load_model(modelFileName)
# loadedModel.summary()

def predictIndividualFile(songName, modelName):
    y, sr = librosa.load(songName, mono=True, duration=30)
    dataRow = []
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    dataRow.append(np.mean(chroma_stft))
    rmse = librosa.feature.rms(y=y)
    dataRow.append(np.mean(rmse))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    dataRow.append(np.mean(spec_cent))
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    dataRow.append(np.mean(spec_bw))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    dataRow.append(np.mean(rolloff))
    zcr = librosa.feature.zero_crossing_rate(y)
    dataRow.append(np.mean(zcr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for e in mfcc:
        dataRow.append(np.mean(e))
    dataRowNumpyArray = np.array(dataRow)
    # scaler = StandardScaler()
    # cols = scaler.fit_transform([dataRowNumpyArray])
    # featureColsScaled = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    model = load_model(modelName)
    model.summary()
    # predicted = model.predict(cols)
    predicted = model.predict(np.array([dataRowNumpyArray, ]))
    print(predicted)
    print(f'predicted class: {np.argmax(predicted[0])}')
    return predicted, np.argmax(predicted[0])

def predictGenre(genreName):
    vals = {}
    for i in range(90, 100):
        songName =  f'genres/{genreName}/{genreName}.000{i}.wav'
        predictedClass = predictIndividualFile(songName, modelFileName)[1]
        if predictedClass in vals:
            vals[predictedClass] += 1
        else:
            vals[predictedClass] = 1
    print(f'predicted genres: {vals}')
    return vals

# predictGenre("rock")
testFile = "sample_audio/Muddy Waters - Hoochie Coochie Man   ( Chess 1954).wav"
predictedClass = predictIndividualFile(testFile, modelFileName)[1]
print(f'predicted class: {predictedClass}')


