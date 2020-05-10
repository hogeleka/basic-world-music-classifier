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


modelName = "world_music_region_model_4_layers_500epochs.h5"
model = load_model(modelName)
# model.summary()


##Old method  -- keep commented out ----
# def predictRegion(audioFilePath):
#     y, sr = librosa.load(audioFilePath, mono=True, duration=60)
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
#     predicted = model.predict(np.array([dataRowNumpyArray, ]))
#     print(predicted)
#     print(f'predicted class: {np.argmax(predicted[0])}')
#     return np.argmax(predicted[0])

def rankList(scores):
    """takes an array of numbers and returns an array of the relative positions of the indices when values are sorted in reverse order
    For example: [5.2, 7.1, 2.9, 1.2] --> [2, 1, 3, 4] """
    vals = {}
    for i in range(len(scores)):
        vals[i] = scores[i]
    output = [None] * 4  #our project has 4 different classifications
    sortedRanks = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)}
    scoreFactor = 0
    for key in sortedRanks:
        output[key] = 4 - scoreFactor
        scoreFactor += 1
    return output


def predictRegionOffset(audioFilePath, offsetValue):
    y, sr = librosa.load(audioFilePath, offset=offsetValue, mono=True, duration=60)
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
    predicted = model.predict(np.array([dataRowNumpyArray, ]))
    print(predicted)
    predictedClass = np.argmax(predicted[0])
    print(f'predicted class: {predictedClass}')
    return predicted[0], predictedClass


def predictSong(audioPath):
    """
    Takes an audio track of duration floor(x) minutes,
    breaks it down into x sections,
    analyse each one minute chunk
    process the results of all 1 minute ch
    :param audioPath:
    :return: 2 sorted lists reflecting relative rankings based on 2 different methodologies of processing results
    """
    start = time.time()
    audioLength = int(librosa.get_duration(filename=audioPath))
    numOneMinuteChunks = audioLength // 60 #determine how many complete 1 minute chunks we need
    classPredictions = {0:0, 1:0, 2:0, 3:0}#initialise the datastructure to keep count of predictions
    firstPositionPredictions = {0:0, 1:0, 2:0, 3:0}
    for i in range(numOneMinuteChunks):
        offsetValue = i * 60
        prediction, top = predictRegionOffset(audioFilePath=audioPath, offsetValue=offsetValue)
        scores = rankList(prediction)
        for i in range(len(scores)):
            classPredictions[i] += scores[i]
        firstPositionPredictions[top] += 1
    sortedPredictions = {k: v for k, v in sorted(classPredictions.items(), key=lambda item: item[1], reverse=True)}
    print(sortedPredictions)
    sortedFirstPositionPredictions = {k: v for k,v in sorted(firstPositionPredictions.items(), key=lambda item: item[1], reverse=True)}
    print(sortedFirstPositionPredictions)
    end = time.time()
    print(f'predicted song in {end-start} seconds')
    return sortedPredictions, sortedFirstPositionPredictions

songName = "sample_audio/Boubacar Traoré & Ali Farka Touré - Duna Ma Yelema.wav"
predictSong(songName)
