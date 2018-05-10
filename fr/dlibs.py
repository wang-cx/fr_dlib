# -*- coding: utf-8 -*-
import dlib
import numpy as np

from . import utils

face_detector = dlib.cnn_face_detection_model_v1(
    "/Users/administrator/workspace/AI_models/dlib/mmod_human_face_detector.dat")

face_shape_predictor = dlib.shape_predictor(
    "/Users/administrator/workspace/AI_models/dlib/shape_predictor_5_face_landmarks.dat")

face_descriptor = dlib.face_recognition_model_v1(
    "/Users/administrator/workspace/AI_models/dlib/dlib_face_recognition_resnet_model_v1.dat")


def detect_faces(im):
    faces = face_detector(im)
    return utils.to_dlib_rectangles(faces)


def compute_face_descriptor(im, face):
    shape = face_shape_predictor(im, face)
    descriptor = face_descriptor.compute_face_descriptor(im, shape, 100)
    return np.array(descriptor)
