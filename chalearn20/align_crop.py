import numpy as np
from skimage import transform
from skimage import img_as_ubyte
import paths
import os
from scipy import ndimage
import dlib
import cv2


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_template_landmark():
    file_path = paths.TEMPLATE
    template = list(np.genfromtxt(file_path, dtype=str))
    num_landmarks = len(template)
    template_arr = np.zeros((num_landmarks, 2), dtype='int')
    for i in range(num_landmarks):
        x, y = template[i].strip().split(',')
        template_arr[i] = [int(x), int(y)]
    return template_arr


class FaceAligner:
    def __init__(self, predictor, desiredFaceWidth, desiredLeftEye=(0.35, 0.35), desiredFaceHeight=None):
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align_to_template_similarity(self, image, gray, rect):
        template_landmarks = get_template_landmark()
        detected_landmarks = shape_to_np(self.predictor(gray, rect))

        tf = transform.estimate_transform('similarity', detected_landmarks, template_landmarks)
        result = img_as_ubyte(transform.warp(image, inverse_map=tf.inverse, output_shape=(self.desiredFaceWidth,
                                                                                          self.desiredFaceWidth, 3)))
        return result


def find_largest_face(face_rectangles):
    number_rectangles = len(face_rectangles)

    if number_rectangles == 0:
        return None
    elif number_rectangles == 1:
        return face_rectangles[0]
    else:
        largest = 0
        which_rectangle = None
        for i in range(number_rectangles):
            r = face_rectangles[i]
            # it's a square so only one side needs to be checked
            width = r.right() - r.left()
            if width > largest:
                largest = width
                which_rectangle = i
        # print('rectangle %d is largest with a side of %d' % (which_rectangle, largest))
        return face_rectangles[which_rectangle]


def align_face(image, xp):
    predictor = paths.PREDICTOR
    predictor = dlib.shape_predictor(predictor)
    detector = dlib.get_frontal_face_detector()
    fa = FaceAligner(predictor, desiredFaceWidth=196)  # 208?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2)
    if len(face_rectangles) == 0:
        print('no face detected in the generated image')
        return xp.zeros((image.shape), dtype=xp.uint8)
    largest_face_rectangle = find_largest_face(face_rectangles)
    face_aligned = fa.align_to_template_similarity(image, gray, largest_face_rectangle)
    return face_aligned.astype(xp.uint8)
