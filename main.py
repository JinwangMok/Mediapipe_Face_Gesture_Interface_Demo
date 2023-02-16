import cv2
import numpy as np
import logging
from utilized_face_mesh import Utilized_face_mesh
from angles_of_face import Angles_of_face
from _utils import *


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
    "right_eye_tip" : 33,
    "between_both_eyes" : 168
}


def add_stream_handler_with_formatter(logger, formatter):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def erase_previous_log_file(filename="./log.txt"):
    f = open(filename, 'w')
    f.write("")
    f.close()

def add_file_handler_with_formatter(logger, formatter, filename="./log.txt"):
    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    add_stream_handler_with_formatter(logger, formatter)
    erase_previous_log_file()
    add_file_handler_with_formatter(logger, formatter)
    
    logger.info("Set up logger")


if __name__ == "__main__":
    logger = logging.getLogger()
    setup_logger(logger)

    cap = cv2.VideoCapture(CAM_NUM)

    face_mesh = Utilized_face_mesh(**FACE_MESH_PARAMS)
    with face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          logger.warning("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        coordinates_of_multi_face = face_mesh.get_3D_coordinates_from_bgr_image(image, TARGET_FACE_LANDMARKS)
        if len(coordinates_of_multi_face) == 0:
            logger.info("No face detected.")
            continue
        angle_of_single_face = Angles_of_face(coordinates_of_multi_face['0'])
        face_mesh.paint_last_landmarks_to_image(image)
        
        # Test ###
        
        # TODO: 각도 계산 후 저장(향후 각도와 거리(z)를 이용하여 커서의 위치를 표현할 예정). depth는 음수일수록 카메라와 가까움을 의미
        image = cv2.flip(image, 1)
        # l2n_dist = L2distnace(coordinates['0']['left_eye_tip'][0], coordinates['0']['left_eye_tip'][1], coordinates['0']['nose_tip'][0], coordinates['0']['nose_tip'][1])
        # r2n_dist = L2distnace(coordinates['0']['right_eye_tip'][0], coordinates['0']['right_eye_tip'][1], coordinates['0']['nose_tip'][0], coordinates['0']['nose_tip'][1])
        # if np.abs(l2n_dist - r2n_dist) > 30:
        #     cv2.putText(image, "Panning!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(image, f"leftEye2nose: {l2n_dist}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, f"rightEye2nose: {r2n_dist}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"rolling_angle: {angle_of_single_face.rolling_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"tilting_angle: {angle_of_single_face.tilting_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"panning_angle: {angle_of_single_face.panning_angle}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)

        # cv2.putText(image, f"tilting_angle: {angle_of_single_face.tilting_angle}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 1, cv2.LINE_AA)
        ###

        cv2.imshow('MediaPipe Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()