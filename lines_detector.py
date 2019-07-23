import cv2
import numpy as np

class VConvolutionFilter(object):
   """A filter that applies a convolution to V (or all of
   BGR)."""

   def __init__(self, kernel):
       self._kernel = kernel

   def apply(self, src, dst):
       """Apply the filter with a BGR or gray
       source/destination."""
       cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
   """A sharpen filter with a 1-pixel radius."""

   def __init__(self):
       kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
       VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VConvolutionFilter):
   """An edge-finding filter with a 1-pixel radius."""

   def __init__(self):
       kernel = numpy.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])
       VConvolutionFilter.__init__(self, kernel)

class BlurFilter(VConvolutionFilter):
   """A blur filter with a 2-pixel radius."""

   def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                             [0.04, 0.04, 0.04, 0.04, 0.04],
                             [0.04, 0.04, 0.04, 0.04, 0.04],
                             [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)

class EmbossFilter(VConvolutionFilter):
   """An emboss filter with a 1-pixel radius."""

   def __init__(self):
       kernel = np.array([[-2, -1, 0],
                             [-1, 1, 1],
                             [ 0, 1, 2]])
       VConvolutionFilter.__init__(self, kernel)

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
   if blurKsize >= 3:
       blurredSrc = cv2.medianBlur(src, blurKsize)
       graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
   else:
       graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
   cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
   normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
   channels = cv2.split(src)
   for channel in channels:
       channel[:] = channel * normalizedInverseAlpha
   cv2.merge(channels, dst)


   kernels = [np.array([[-1, 9, -1],
                      [9, 9, 9],
                      [-1, 9, -1]]),
          np.array([[-1, -1, -1],
                      [9, 9, 0],
                      [-1, -1, -1]]),
          np.array([[-1, 9, -1],
                      [-1, 9, -1],
                      [-1, 9, -1]])]
   for kernel in kernels:
        cv2.filter2D(src, -1, kernel, dst)

img = cv2.imread("image/image2.jpg", 0)
height, width, channels = img.shape
cv2.imwrite("image/image2_.jpg", cv2.Canny(img, 200, 300))

img = np.zeros((height, width), dtype=np.uint8)
img[50:150, 50:150] = 255

ret, thresh = cv2.threshold(img, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()