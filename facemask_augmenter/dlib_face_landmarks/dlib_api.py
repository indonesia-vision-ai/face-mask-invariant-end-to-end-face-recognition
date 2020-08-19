from pathlib import Path

import dlib
import numpy as np


predictor_68_point_model = str(Path(__file__).parent / "shape_predictor_68_face_landmarks.dat")
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = str(Path(__file__).parent / "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    return cnn_face_detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image):
    face_locations_ = _raw_face_locations(face_image)
    face_locations_ = [rectangle.rect for rectangle in face_locations_]

    return [pose_predictor_68_point(face_image, face_location) for face_location in face_locations_]


def get_face_landmarks(face_image):
    landmarks = _raw_face_landmarks(face_image)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]
