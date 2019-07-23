import numpy as np
import cv2
D2 = 512
D1 = 270

img = cv2.imread('Preprocessed/1/2019_04_11__15_55_03_53__5319_img_1.jpg',0)

img = cv2.resize(img, (D2, D1))
center = (D2/1.5,D1/2)
scale = 1.0

for angle in np.arange(45,360,45):

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (D2, D1), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite("Preprocessed/1/2019_04_11__15_55_03_53__5319_img_1"+str(angle)+".jpg", rotated)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
cv2.imshow("image2", img)
