# import the necessary packages
import numpy as np
import argparse
import cv2
import os
path = "Datasets/3/"
output_path = "Preprocessed/3/"
for filename in os.listdir(path):
    image = cv2.imread(path+filename)


    # hsv hue sat value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([100,0,0])
    upper_green = np.array([150,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)

    #cv2.imwrite(filename, result)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
    #break
    cv2.imwrite(output_path+filename, result)
