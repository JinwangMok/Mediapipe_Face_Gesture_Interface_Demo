import cv2
import numpy as np
import logging
from logger import logger

from utilized_face_mesh import Utilized_face_mesh
from face_info_per_frame import Face_info_per_frame
from buffer_for_face_info import Buffer_for_face_info
from _utils import is_empty


CAM_NUM = 1
WIDTH = 1280
HEIGHT = 720

FACE_MESH_PARAMS = {
    "max_num_faces" : 1,
    "refine_landmarks" : True,
    "min_detection_confidence" : 0.5,
    "min_tracking_confidence" : 0.5
}

TARGET_FACE_LANDMARKS = {
    "nose_tip" :      1,
    "left_eye_tip" :  263,
    "left_eye_upper" : 386,
    "left_eye_under" :  374,
    "right_eye_tip" : 33,
    "right_eye_upper" : 159,
    "right_eye_under" : 145,
}


if __name__ == "__main__":
    cap = cv2.VideoCapture(CAM_NUM)
    face_mesh = Utilized_face_mesh(**FACE_MESH_PARAMS)
    buffer_for_face_info = Buffer_for_face_info(max_size=100)
    with face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          logger.warning("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        coordinates_of_multi_face = face_mesh.get_3D_coordinates_from_bgr_image(image, TARGET_FACE_LANDMARKS)
        if is_empty(coordinates_of_multi_face):
            logger.info("No face detected.")
            continue

        face_info_per_frame = Face_info_per_frame(coordinates_of_multi_face['0'])
        buffer_for_face_info.push_back(face_info_per_frame)

        face_mesh.paint_last_landmarks_to_image(image)
        
        # Test ###
        image = cv2.flip(image, 1)
        
        cv2.putText(image, f"eye_state_upper: {face_info_per_frame.left_eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"eye_state_under: {face_info_per_frame.right_eye_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        # cv2.putText(image, f"rolling_angle: {face_info_per_frame.rolling_angle}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        # cv2.putText(image, f"tilting_angle: {face_info_per_frame.tilting_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        # cv2.putText(image, f"panning_angle: {face_info_per_frame.panning_angle}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)

        ###

        cv2.imshow('MediaPipe Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()