import cv2
import numpy as np
import logging
from utilized_face_mesh import utilized_face_mesh
import _utils


CAM_NUM  = 1
WIDTH    = 1280
HEIGHT   = 720

FACE_MESH_PARAMS = {
    "max_num_faces" : 1,
    "refine_landmarks" : True,
    "min_detection_confidence" : 0.5,
    "min_tracking_confidence" : 0.5
}

TARGET_FACE_LANDMARKS = {
    "nose_tip" :      1,
    "left_eye_tip" :  263,
    "right_eye_tip" : 33
}


logger = logging.getLogger()
def setup_logger():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Set up logger")
setup_logger()


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    face_mesh = utilized_face_mesh(**FACE_MESH_PARAMS)

    with face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          logger.warning("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        coordinates = face_mesh.get_3D_coordinates_from_bgr_image(image, TARGET_FACE_LANDMARKS)

        face_mesh.paint_last_landmarks_to_image(image)
            
        #     # Customized⭐️
        #     image = cv2.flip(image, 1)
        #     l2n_dist = l2distnace(left_eye_x, left_eye_y, nose_x, nose_y)
        #     r2n_dist = l2distnace(right_eye_x, right_eye_y, nose_x, nose_y)
        #     if np.abs(l2n_dist - r2n_dist) > 30:
        #       cv2.putText(image, "Panning!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 1, cv2.LINE_AA)
        #     cv2.putText(image, f"leftEye2nose: {l2n_dist}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 1, cv2.LINE_AA)
        #     cv2.putText(image, f"rightEye2nose: {r2n_dist}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 1, cv2.LINE_AA)
        #     cv2.putText(image, f"nose_z: {nose_z}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
          break

    cap.release()