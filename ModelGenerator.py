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


csvDataFileName = "world_music_data.csv"
fileDurationInSeconds = 60


### --------------uncomment and modify code chunk below up until line 83 to create your csv file of your model----####
#prepare csv file header

# header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()
#
#
#
# #extracting every audio
#
# # write header row to file
# start0 = time.time()
# file = open(csvDataFileName, 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
# #
# # genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
#
# wm_regions = 'latin_america_carrib south_asia south_east_asia sub_sahara_africa'.split()
#
# # def extractAudioSpecs():
# for g in wm_regions:
#     # pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
#     for filename in os.listdir(f'wm_regions/{g}'):
#         songname = f'wm_regions/{g}/{filename}'
#         # trackNumber = int(songname[-6:-4])
#         print(f'---reading {g} track {filename}------')
#         y, sr = librosa.load(songname, mono=True, duration=fileDurationInSeconds)
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         rmse = librosa.feature.rms(y=y)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         filenameForCSV = filename.replace(" ", "_")
#         to_append = f'{filenameForCSV} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {g}'
#         file = open(csvDataFileName, 'a', newline='')
#         with file:
#             writer = csv.writer(file)
#             writer.writerow(to_append.split())
#
#     print(f'----done with {g}-------')
#
# end0 = time.time()
# print(f'----finished writing csv in {end0 - start0} seconds------')
###-----STOP here. Code up to this point helps you write csv output for training ----#



###---Uncommment chunk below up until line 144 to actually read csv data, train model and save .h5 file -----####
# start1 = time.time()
# data = pd.read_csv(csvDataFileName)
# data.head()
#
# print(data.shape)
#
# # Dropping unneccesary columns
# data = data.drop(['filename'],axis=1)
#
#
# #Encoding the labels
# genre_list = data.iloc[:, -1]
# encoder = LabelEncoder()
# y = encoder.fit_transform(genre_list)
#
# #Scaling Feature Columns
# scaler = StandardScaler()
# X = np.array(data.iloc[:, :-1], dtype=float)
# # X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
#
# #Divide into test and train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# end1 = time.time()
# print(f'------finished reading, processing, and dividing up data in {end1-start1} seconds--------')
#
# start2 = time.time()
#
# #building network
# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#
# model.add(layers.Dense(128, activation='relu'))
#
# model.add(layers.Dense(64, activation='relu'))
#
# model.add(layers.Dense(4, activation='softmax'))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# epochs = 500
# history = model.fit(X_train,
#                     y_train,
#                     epochs=epochs,
#                     batch_size=128)
#
# test_loss, test_acc = model.evaluate(X_test,y_test)
# print('test_acc: ',test_acc)
# end2 = time.time()
# print(f'------------finished with epochs and stuff in {end2-start2}seconds -------------')
#
# model.summary()
# modelFileName = f'world_music_region_model_4_layers_{epochs}epochs.h5'
# modelSaved = model.save(modelFileName)


