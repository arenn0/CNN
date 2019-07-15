from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
import EGP
import numpy as np
import cv2

import matplotlib.pyplot as plt

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


batch_size = 64
epochs = 10

D1 = EGP.D1
D2 = EGP.D2
C = 3


(X_train, y_train), (X_test, y_test) = EGP.load_data("Preprocessed/")
N_train = len(X_train)
N_test = len(X_test)
N_classes = max(y_train)+1
print("Data Loaded")

X_train = (np.array(X_train, dtype="float")/255.0).reshape(N_train, D1, D2, C)
X_test = (np.array(X_test, dtype="float")/255.0).reshape(N_test, D1, D2, C)

# plt.imshow(X_train[10])
#plt.show()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, activation='relu', strides=(1,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(N_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

