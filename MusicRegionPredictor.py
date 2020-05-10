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


# modelName = "world_music_region_model_500epocs.h5"
modelName = "world_music_region_model_4_layers_500epochs.h5"
model = load_model(modelName)
# model.summary()

def predictRegion(audioFilePath):
    y, sr = librosa.load(audioFilePath, mono=True, duration=60)
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
    # model.summary()
    predicted = model.predict(np.array([dataRowNumpyArray, ]))
    print(predicted)
    print(f'predicted class: {np.argmax(predicted[0])}')
    return np.argmax(predicted[0])
    # return str(np.argmax(predicted[0]))



# for i in range(3):
#     fileName = f'sample_audio/Javanese Gamelan Ensemble - Pelog Barang - Singa Nebah (The Pouncing Lion)_{i}.wav'
#     predictRegion(fileName)

# for i in range(4):
#     fileName = f'sample_audio/Jalikunda African Drums take the Montserrat African Music Festival by storm_{i}.wav'
#     predictRegion(fileName)

# predictRegion("wm_regions/sub_sahara_africa/Angola Ngoma Dance Ensemble_00.wav")

# predictions = {}
# for i in range(7):
#     fileName = f"sample_audio/Kpanlogo (DMC)_{i}.wav"
#     predictedClass = predictRegion(fileName)
#     if predictedClass in predictions:
#         predictions[predictedClass] += 1
#     else:
#         predictions[predictedClass] = 1
# print(predictions)

def rankList(scores):
    vals = {}
    for i in range(len(scores)):
        vals[i] = scores[i]
    output = [None] * 4
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
    # model.summary()
    predicted = model.predict(np.array([dataRowNumpyArray, ]))
    print(predicted)
    predictedClass = np.argmax(predicted[0])
    print(f'predicted class: {predictedClass}')
    # print(f'predicted class: {np.argmax(predicted[0])}')
    return predicted[0], predictedClass
    # return np.argmax(predicted[0])


def predictSong(audioPath):
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
    print(firstPositionPredictions)
    end = time.time()
    print(f'predicted song in {end-start} seconds')
    return sortedPredictions