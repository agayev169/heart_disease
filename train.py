#The best val_acc = 0.8456

import csv
import numpy as np

def prepare_data(data):
	X, y = [], []
	try:
		for i in range(13):
			X.append(float(data[i]))
		y.append(float(data[13]))
	except:
		return X, y
	return X, y


f = open('data.csv')
r = csv.reader(f)
X, y = [], []
for row in r:
	X_r, y_r = prepare_data(row)
	if not X_r == [] and len(X_r) == 13:
		X.append(X_r)
		y.append(y_r)

f.close()

X = np.array(X)
y = np.array(y)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize, to_categorical
import time


X = normalize(X)
y = normalize(y)


model = Sequential()

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))

adam = Adam(lr = 0.0005)
model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics = ['accuracy'])
model.fit(X, y, epochs = 1000, batch_size = 1, validation_split = 0.5)