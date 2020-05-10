import numpy as np
import time
import pandas as pd

# Pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
from keras import models
from keras import layers

csvDataFileName = "../csv/world_music_data.csv"

start1 = time.time()
data = pd.read_csv(csvDataFileName)
data.head()

print(data.shape)

# Dropping unnecessary columns
data = data.drop(['filename'], axis=1)


# Encoding the labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# Scaling Feature Columns
scaler = StandardScaler()
X = np.array(data.iloc[:, :-1], dtype=float)
# X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

# Divide into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
end1 = time.time()
print(f'------finished reading, processing, and dividing up data in {end1-start1} seconds--------')

start2 = time.time()

# building network
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 500
history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=128)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)
end2 = time.time()
print(f'------------finished with epochs and stuff in {end2-start2}seconds -------------')

model.summary()
modelFileName = f'../model/world_music_region_model_4_layers_{epochs}epochs.h5'
modelSaved = model.save(modelFileName)
