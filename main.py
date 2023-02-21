import cv2
import logging
from logger import logger

from utilized_face_mesh import Utilized_face_mesh
from face_gesture_processor import Face_gesture_processor
from painter import Painter


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
        processor = Face_gesture_processor(face_mesh, WIDTH, HEIGHT)
        painter = Painter(face_mesh, WIDTH, HEIGHT)
        while cap.isOpened():
            success, camera_frame = cap.read()
            if not success:
                logger.warning("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            success, result = processor.process(camera_frame, TARGET_FACE_LANDMARKS)
            if not success:
                logger.info("No face detected.")
                continue
            gesture, cursor_position = result

            painter.paint_canvas(gesture, cursor_position)
 
            painter.paint_detail(processor, camera_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()