import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple


class Utilized_face_mesh(mp.solutions.face_mesh.FaceMesh):
    def __init__(self, **FACE_MESH_PARAMS):
        super().__init__(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.__painter = mp.solutions.drawing_utils
        self.__paint_style = mp.solutions.drawing_styles
        self.__painter_drawing_spec = self.__painter.DrawingSpec(thickness=1, circle_radius=1)
        self.__last_landmarks_of_faces: Dict = {}
        self.__last_image = None


    def get_3D_coordinates_from_bgr_image(
        self, 
        img,
        target_landmarks:Dict[str, int]
        ) -> Dict[str, Dict[str, Tuple[np.float64]]]:
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        img.flags.writeable = False

        parsed_results = self.__parse_bgr_image(img)
        
        coordinates_of_multi_face:Dict[str, Dict] = {}
        if parsed_results.multi_face_landmarks:
            for face_num, face_landmarks in enumerate(parsed_results.multi_face_landmarks):
                self.__last_landmarks_of_faces[str(face_num)] = face_landmarks
                
                coordinates_of_multi_face[str(face_num)]:Dict[str, Tuple[np.float64]] = {}
                for landmark_name, landmark_num in target_landmarks.items():
                    x, y, z = self.__get_coordinate_from_landmark(face_landmarks.landmark, landmark_num)
                    coordinates_of_multi_face[str(face_num)][landmark_name] = (x, y, z)

        img.flags.writeable = True

        return coordinates_of_multi_face


    def __parse_bgr_image(self, img:cv2.Mat):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.process(img)
        
        return results


    def __get_coordinate_from_landmark(
        self,
        landmarks,
        num:int,
        ) -> Tuple[np.float64]:
        x = np.float64(landmarks[num].x)     #[0, 1]
        y = np.float64(landmarks[num].y)     #[0, 1]
        z = np.float64(landmarks[num].z + 1) #[-1, 1] -> [0, 2]

        return x, y, z
    

    def paint_last_landmarks_to_image(self, img):
        for face_landmarks in self.__last_landmarks_of_faces.values():
            self.__painter.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.__paint_style.get_default_face_mesh_tesselation_style()
            )
            self.__painter.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.__paint_style.get_default_face_mesh_contours_style()
            )
            self.__painter.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.__paint_style.get_default_face_mesh_iris_connections_style()
            )