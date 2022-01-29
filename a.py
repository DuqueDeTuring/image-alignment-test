import cv2
import numpy

im1 = cv2.imread('1.jpg')
im2 = cv2.imread('2.jpg')

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

sz = im2.shape

warp_mode = cv2.MOTION_HOMOGRAPHY #EUCLIDEAN

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    warp_matrix = numpy.eye(3,3,dtype=numpy.float32)
else:
    warp_matrix = numpy.eye(2,3,dtype=numpy.float32)

number_of_iterations = 1000

termination_eps = 1e-10

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

(cc, warp_matrix) = cv2.findTransformECC(im2_gray, im1_gray, warp_matrix, warp_mode, criteria, None, 5)

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    im2_aligned = cv2.warpPerspective(im1, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
    im2_aligned = cv2.warpAffine(im1, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR +cv2.WARP_INVERSE_MAP)

cv2.imwrite("3.jpg", im2_aligned)
