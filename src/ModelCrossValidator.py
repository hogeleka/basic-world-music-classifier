import pandas as pd
import numpy as np
import time

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
from keras import models
from keras import layers

import warnings
warnings.filterwarnings('ignore')

csvDataFileName = "../csv/world_music_data.csv"
fileDurationInSeconds = 60
epochs = 500

seed = 5
np.random.seed(seed)

start1 = time.time()
data = pd.read_csv(csvDataFileName)
data.head()

print(data.shape)

# Dropping unnecessary columns
data = data.drop(['filename'],axis=1)

# Encoding the labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# Scaling Feature Columns
scaler = StandardScaler()
X = np.array(data.iloc[:, :-1], dtype=float)

# Divide into test and train
end1 = time.time()
print(f'------finished reading, processing, and dividing up data in {end1-start1} seconds--------')

start2 = time.time()

# building network
kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kFold.split(X, y):

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X[train].shape[1],)))

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X[train],
                        y[train],
                        epochs=epochs,
                        batch_size=128, verbose=0)

    test_loss, test_acc = model.evaluate(X[test],y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], test_acc * 100))
    cvscores.append(test_acc * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end2 = time.time()
print(f'------------finished with epochs and stuff in {end2-start2}seconds -------------')
