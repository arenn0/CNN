from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
from keras.regularizers import l2
from keras.optimizers import adam, sgd
from keras import backend as K
import EGP
import numpy as np
import cv2

import matplotlib.pyplot as plt


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


batch_size = 64
epochs = 50

D1 = EGP.D1
D2 = EGP.D2
C = EGP.C


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
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', strides=(1,1), kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(800, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dense(N_classes, activation='softmax', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))

model.compile(loss=categorical_crossentropy, optimizer=adam(lr=0.000001), metrics=['accuracy',f1_m,precision_m, recall_m])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test), shuffle=True)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
predictions = model.predict_classes(X_test).reshape((-1,))
for i in predictions:
    if predictions[i] != np.argmax(y_test[i]):
        print("predicted: " + str(predictions[i]), "true: ", str(y_test[i]))
# Accuracy 0.84 on Preprocessed
# Accuracy 0.77 on Dataset

