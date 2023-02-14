import cv2
import mediapipe as mp
from typing import Dict, Tuple

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class utilized_face_mesh(mp.solutions.face_mesh.FaceMesh):
    def __init__(self, **FACE_MESH_PARAMS):
        super().__init__(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.painter = mp.solutions.drawing_utils
        self.paint_style = mp.solutions.drawing_styles
        self.last_landmarks_of_faces: Dict = {}
        self.last_image = None


    def get_3D_coordinates_from_bgr_image(
        self, 
        img,
        target_landmarks:Dict[str, int]
        ) -> Dict[str, Dict[str, Tuple[float]]]:
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        img.flags.writeable = False

        parsed_results = self.parse_bgr_image(img)
        
        coordinates: Dict[str, Dict] = {}
        if parsed_results.multi_face_landmarks:
            for face_num, face_landmarks in enumerate(parsed_results.multi_face_landmarks):
                self.last_landmarks_of_faces[str(face_num)] = face_landmarks
                
                coordinates[str(face_num)]: Dict[str, Tuple[float]] = {}
                for landmark_name, landmark_num in target_landmarks.items():
                    x, y, z = self.get_coordinate_from_landmark(face_landmarks.landmark, landmark_num)
                    coordinates[str(face_num)][landmark_name] = (x, y, z)

        img.flags.writeable = True

        return coordinates


    def parse_bgr_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.process(img)

        return results


    def get_coordinate_from_landmark(
        self,
        landmarks,
        num:int,
        ) -> Tuple[int, int, int]:
        x = landmarks[num].x
        y = landmarks[num].y
        z = landmarks[num].z

        return x, y, z
    

    def paint_last_landmarks_to_image(self, img):
        for face_landmarks in self.last_landmarks_of_faces.values():
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.paint_style.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.paint_style.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.paint_style.get_default_face_mesh_iris_connections_style())