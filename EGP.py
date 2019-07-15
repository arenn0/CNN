from keras.preprocessing import image
import os
import numpy as np
import random
import cv2

D2 = int(4096/8)
D1 = 270

def load_data(path): # -> (X_train, y_train), (X_test, y_test)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(4):
        for filename in os.listdir(path+str(i)+"/"):
            img = cv2.imread(path + str(i) + "/" + filename)
            img = cv2.resize(img, (D2, D1)).flatten()
            if random.randint(1,11) == 1:
                X_test.append(img)
                y_test.append(i)

            else:
                X_train.append(img)
                y_train.append(i)

    return (X_train, y_train), (X_test, y_test)
