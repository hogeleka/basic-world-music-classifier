# import librosa
# import librosa.display
import librosa.feature
import librosa.core.audio
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

# class MusicRegionPredictor():
#
#     def __init__(self):
#
#         self.model = load_model()
#         pass
#
#     def predictRegion(self, audioFilePath):
#         modelFileName = "musicGenreModel_1000New.h5"
#         y, sr = librosa.load(audioFilePath, mono=True, duration=30)
#         dataRow = []
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         dataRow.append(np.mean(chroma_stft))
#         rmse = librosa.feature.rms(y=y)
#         dataRow.append(np.mean(rmse))
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         dataRow.append(np.mean(spec_cent))
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         dataRow.append(np.mean(spec_bw))
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         dataRow.append(np.mean(rolloff))
#         zcr = librosa.feature.zero_crossing_rate(y)
#         dataRow.append(np.mean(zcr))
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         for e in mfcc:
#             dataRow.append(np.mean(e))
#         dataRowNumpyArray = np.array(dataRow)
#         model = load_model(modelFileName)
#         model.summary()
#         predicted = model.predict(np.array([dataRowNumpyArray, ]))
#         print(predicted)
#         print(f'predicted class: {np.argmax(predicted[0])}')
#         return str(np.argmax(predicted[0]))

        # return predicted, np.argmax(predicted[0])
modelName = "musicGenreModel_1000New.h5"
model = load_model(modelName)
model.summary()

def predictRegion(audioFilePath):
    # modelFileName = "musicGenreModel_1000New.h5"
    y, sr = librosa.load(audioFilePath, mono=True, duration=30)
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
    # model = load_model(modelFileName)
    model.summary()
    predicted = model.predict(np.array([dataRowNumpyArray, ]))
    print(predicted)
    print(f'predicted class: {np.argmax(predicted[0])}')
    return str(np.argmax(predicted[0]))

