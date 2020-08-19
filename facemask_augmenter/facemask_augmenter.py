from pathlib import Path
import glob
import random

import cv2
import numpy as np
from .dlib_face_landmarks import get_face_landmarks


class FacemaskAugmenter:
    def __init__(self, facemask_paths=None):
        if facemask_paths is None:
            facemask_glob_pattern = str(Path(__file__).parent / "facemask-images/*.png")
            self.facemask_paths = glob.glob(facemask_glob_pattern)
        else:
            self.facemask_paths = facemask_paths


    def augment_faces(self, image: np.ndarray) -> np.ndarray:
        face_landmarks_list = get_face_landmarks(image)

        for face_landmarks in face_landmarks_list:
            image = self._augment(image, face_landmarks)

        return image


    def _augment(self, image: np.ndarray, face_landmarks: dict) -> np.ndarray:
        img_face = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        left_upper_chin = np.array(face_landmarks['chin'][0])
        left_lower_chin = np.array(face_landmarks['chin'][5])
        right_upper_chin = np.array(face_landmarks['chin'][16])
        right_lower_chin = np.array(face_landmarks['chin'][11])
        nose_upper = np.array(face_landmarks['nose_bridge'][1])
        center_chin = np.array(face_landmarks['chin'][8])

        facemask = cv2.imread(random.choice(self.facemask_paths), -1)
        w_split = int(facemask.shape[1] / 2)

        facemask_left = facemask[:, :w_split]
        facemask_right = facemask[:, w_split:]

        left_h, left_w = facemask_left.shape[:2]
        size_h, size_w = img_face.shape[:2]

        # Warp facemask halves to desired landmark points
        input_pts = np.float32([[0, 0], [left_w, 0],
                                [left_w, left_h],[0, left_h]])

        out_left_pts = np.float32([left_upper_chin, nose_upper,
                                   center_chin, left_lower_chin])
        LM = cv2.getPerspectiveTransform(input_pts, out_left_pts)
        left_out = cv2.warpPerspective(facemask_left, LM, (size_w, size_h),
                                       flags=cv2.INTER_LINEAR)
        left_out_no_alpha = cv2.warpPerspective(facemask_left[:, :, 0:3],
                                                LM, (size_w, size_h),
                                                flags=cv2.INTER_LINEAR)

        out_right_pts = np.float32([nose_upper, right_upper_chin,
                                    right_lower_chin, center_chin])
        RM = cv2.getPerspectiveTransform(input_pts, out_right_pts)
        right_out = cv2.warpPerspective(facemask_right, RM, (size_w, size_h),
                                        flags=cv2.INTER_LINEAR)
        right_out_no_alpha = cv2.warpPerspective(facemask_right[:, :, 0:3],
                                                 RM, (size_w, size_h),
                                                 flags=cv2.INTER_LINEAR)


        left_msk = left_out[:, :, 3]
        left_msk_inv = cv2.bitwise_not(left_msk)

        right_msk = right_out[:, :, 3]
        right_msk_inv = cv2.bitwise_not(right_msk)

        msk_inv = cv2.bitwise_and(left_msk_inv, right_msk_inv)
        out = cv2.bitwise_or(left_out_no_alpha, right_out_no_alpha)

        bg = cv2.bitwise_and(img_face, img_face, mask=msk_inv)

        result = cv2.add(bg, out)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result
