import librosa.feature
import librosa.core.audio
import numpy as np
import time

from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

modelName = "model/world_music_region_model_4_layers_500epochs.h5"
model = load_model(modelName)

def rankList(scores):
    """takes an array of numbers and returns an array of the relative positions of the indices when values are sorted in reverse order
    For example: [5.2, 7.1, 2.9, 1.2] --> [2, 1, 3, 4] """
    vals = {}
    for i in range(len(scores)):
        vals[i] = scores[i]
    output = [None] * 4  # our project has 4 different classifications
    sortedRanks = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)}
    scoreFactor = 0
    for key in sortedRanks:
        output[key] = 4 - scoreFactor
        scoreFactor += 1
    return output


def getResultClassList(sortedDict):
    """takes a dictionary where class codes are sorted by ranks and returns a corresponding list of Region names
    sorted by rank."""
    musicRegionMap = {
        0: "Latin America/Caribbean",
        1: "South Asia",
        2: "South East Asia",
        3: "Sub-Saharan Africa"
    }
    return [musicRegionMap[code] for code in sortedDict]


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
    numOneMinuteChunks = audioLength // 60  # determine how many complete 1 minute chunks we need
    classScores = {0: 0, 1: 0, 2: 0, 3: 0}  # initialise the data structure to keep scores
    firstPositionPredictions = {0: 0, 1: 0, 2: 0, 3: 0}  # initialise the data structure to keep count of predictions

    for i in range(numOneMinuteChunks):
        offsetValue = i * 60
        prediction, top = predictRegionOffset(audioFilePath=audioPath, offsetValue=offsetValue)
        scores = rankList(prediction)
        for i in range(len(scores)):
            classScores[i] += scores[i]
        firstPositionPredictions[top] += 1

    sortedScoreCountPredictions = {k: v for k, v in sorted(classScores.items(), key=lambda item: item[1], reverse=True)}
    print(sortedScoreCountPredictions)
    if numOneMinuteChunks < 2:
        sortedFirstPositionPredictions = sortedScoreCountPredictions.copy()
    else:
        sortedFirstPositionPredictions = {k: v for k, v in sorted(firstPositionPredictions.items(), key=lambda item: item[1], reverse=True)}
    print(sortedFirstPositionPredictions)
    end = time.time()
    print(f'predicted song in {end - start} seconds')

    return getResultClassList(sortedScoreCountPredictions), getResultClassList(sortedFirstPositionPredictions)

