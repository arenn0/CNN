# import the necessary packages
import numpy as np
import argparse
import cv2

# load the image
image = cv2.imread("image.jpg")

# define the list of boundaries
boundaries = [([150, 150, 200], [255, 255, 255])]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # show the images
    cv2.imwrite("image7.jpg", output)
    cv2.waitKey(0)