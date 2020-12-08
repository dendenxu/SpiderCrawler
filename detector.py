import cv2
import dlib
import struct
import numpy as np
import matplotlib.pyplot as plt

from ctypes import *


class FacePoseDetector:

    def __init__(self):

        self.face = CDLL('pm_deps64.dll')
        self.face.face_init(b'haarcascade_frontalface_default.xml', b'data.bin')

        self.dim = c_int()  # c int variable
        self.pdim = pointer(self.dim)  # pointer

        self.trans = None
        self.rot = None


class LandmarkDetector:

    # 68 landmarks
    landmark_dim = 2
    landmark_num = 68
    outerMouthMarkIndices = range(48, 68)  # starts from 0
    innerMouthMarkIndices = range(60, 68)  # starts from 0

    innerTrianglesIndices = [[66, 62, 65],
                             [62, 66, 61],
                             [67, 60, 61],
                             [63, 65, 62],
                             [65, 63, 64],
                             [61, 66, 67]]

    outerTrianglesIndices = [[66, 56, 57],
                             [56, 66, 65],
                             [66, 57, 58],
                             [53, 64, 63],
                             [64, 53, 54],
                             [48, 49, 60],
                             [49, 50, 61],
                             [50, 51, 61],
                             [63, 51, 52],
                             [51, 63, 62],
                             [52, 53, 63],
                             [54, 55, 64],
                             [55, 56, 65],
                             [60, 67, 59],
                             [49, 61, 60],
                             [58, 59, 67],
                             [59, 48, 60],
                             [58, 67, 66],
                             [61, 51, 62],
                             [65, 64, 55]]

    innerTrianglesIndices = np.array(innerTrianglesIndices, dtype=np.int32)
    outerTrianglesIndices = np.array(outerTrianglesIndices, dtype=np.int32)

    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model\\shape_predictor_68_face_landmarks.dat")
        self.landmarks = np.zeros(shape=(self.landmark_num, self.landmark_dim), dtype=np.int32)

    def get_landmarks(self, img):

        # img : BGR format
        # initialize
        self.landmarks = np.zeros(shape=(self.landmark_num, self.landmark_dim), dtype=np.int32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)

        if len(dets) == 1:  # only one face is plausible in your project

            try:
                shape = self.predictor(img, dets[0])
            except:
                return False
            for i, pt in enumerate(shape.parts()):
                self.landmarks[i, 0] = pt.x
                self.landmarks[i, 1] = pt.y

            return True

        return False


class FaceManipulator(FacePoseDetector, LandmarkDetector):

    def __init__(self):
        FacePoseDetector.__init__(self)
        LandmarkDetector.__init__(self)

        self.img = None
        self.detected = False
        self.triangle_indices = self.innerTrianglesIndices

        self.left = None
        self.right = None
        self.up = None
        self.down = None
        # self.detectFace()

    def detect_face(self, ratio=1):

        if self.get_landmarks(self.img):
            self.detected = True
            self.triangle_indices = np.concatenate([self.outerTrianglesIndices, self.innerTrianglesIndices], axis=0)
            height = abs(self.landmarks[self.triangle_indices[0][0], 1]-self.landmarks[self.triangle_indices[21][0], 1])
            width = abs(self.landmarks[self.triangle_indices[5][0], 0]-self.landmarks[self.triangle_indices[11][0], 0])

            # FIXME Consider twice on this
            # you may not detect thing on the original image
            # so will it be necessary to check for valid mouth width?
            if width < 256//ratio:
                return False
            elif height < width*0.14:
                return False
            return True
        return False

    def get_edge(self):
        # self.img = self.img[y:y+h, x:x+w]
        # landmarks are defined so that x is stored in  [...,0]
        # landmarks are defined so that y is stored in  [...,1]
        self.left = min(self.landmarks[0:-1, 0])
        self.right = max(self.landmarks[0:-1, 0])
        self.up = min(self.landmarks[0:-1, 1])
        self.down = max(self.landmarks[0:-1, 1])
        self.left -= (self.right - self.left)//2
        self.right += (self.right - self.left)//3
        self.up -= self.down - self.up
        self.down += (self.down - self.up)//4


def plotTriangles(img_in, points=None, triangleIndices=None):
    img = img_in.copy()
    triangle_tmp = triangleIndices
    points_tmp = points
    for i in range(len(triangle_tmp)):
        indices = triangle_tmp[i]
        p1 = (int(points_tmp[indices[0], 0]), int(points_tmp[indices[0], 1]))
        p2 = (int(points_tmp[indices[1], 0]), int(points_tmp[indices[1], 1]))
        p3 = (int(points_tmp[indices[2], 0]), int(points_tmp[indices[2], 1]))
        cv2.line(img, p1, p2, 128, 1)
        cv2.line(img, p2, p3, 128, 1)
        cv2.line(img, p3, p1, 128, 1)
        if(i == 5 or i == 11):
            cv2.circle(img, p1, 3, (0, 0, 255))
    for pp in range(27):
        x = int(points_tmp[pp, 0])
        y = int(points_tmp[pp, 1])
        cv2.circle(img, (x, y), 3, (0, 255, 0))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
