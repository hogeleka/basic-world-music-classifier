import librosa.display
import numpy as np
import os
import csv
import time


def getAudioMetadata(songName):
    fileDurationInSeconds = 60
    y, sr = librosa.load(songName, mono=True, duration=fileDurationInSeconds)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc


csvDataFileName = "../csv/world_music_data.csv"

# prepare csv file header
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

# write header row to file
start = time.time()
file = open(csvDataFileName, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

wm_regions = 'latin_america_carrib south_asia south_east_asia sub_sahara_africa'.split()

for g in wm_regions:
    wm_regions_dir = '../wm_regions/'
    for filename in os.listdir(wm_regions_dir + f'{g}'):
        songname = wm_regions_dir + f'{g}/{filename}'
        print(f'---reading {g} track {filename}------')

        chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc = getAudioMetadata(songname)

        filenameForCSV = filename.replace(" ", "_")
        to_append = f'{filenameForCSV} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'

        file = open(csvDataFileName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

    print(f'----done with {g}-------')

end = time.time()
print(f'----finished writing csv in {end - start} seconds------')
