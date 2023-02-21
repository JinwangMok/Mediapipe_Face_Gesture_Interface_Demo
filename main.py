import cv2
import numpy as np
import logging
from logger import logger

from utilized_face_mesh import Utilized_face_mesh
from face_info_per_frame import Face_info_per_frame
from face_angle_stabilizer import Face_angle_stabilizer
from buffer_for_face_info import Buffer_for_face_info
from gesture_recognizer import Gesture_recognizer
from buffer_for_gesture_info import Buffer_for_gesture_info
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
    with face_mesh:
        showing_result = ""
        face_angle_stabilizer = Face_angle_stabilizer(stabilize_time=3.)
        buffer_for_face_info = Buffer_for_face_info(max_size=100)
        gesture_recognizer = Gesture_recognizer(display_width=WIDTH, display_height=HEIGHT)
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
            stabilized_face_info_per_frame = face_angle_stabilizer.stablize_angles(face_info_per_frame)
            buffer_for_face_info.push_back(stabilized_face_info_per_frame)
            
            result:str = gesture_recognizer.recognize_gesture_from_face_info_buffer(buffer_for_face_info)
            if result != "None gesture recognized.":
                showing_result = result

            face_mesh.paint_last_landmarks_to_image(image)
        
            # Test ###
            image = cv2.flip(image, 1)
            
            cv2.putText(image, f"left_eye_state: {face_info_per_frame.left_eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"right_eye_state: {face_info_per_frame.right_eye_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"rolling_angle: {face_info_per_frame.rolling_angle:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"tilting_angle: {face_info_per_frame.tilting_angle:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"panning_angle: {face_info_per_frame.panning_angle:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"gesture_result: {showing_result}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 3, cv2.LINE_AA)
            # cv2.putText(image, f"accumulated_time: {face_angle_stabilizer.accumulated_time}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            # cv2.putText(image, f"rolling_angle: {face_angle_stabilizer.mean_of_angles['rolling']}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            # cv2.putText(image, f"tilting_angle: {face_angle_stabilizer.mean_of_angles['tilting']}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
            # cv2.putText(image, f"panning_angle: {face_angle_stabilizer.mean_of_angles['panning']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)

            ###

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()