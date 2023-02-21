import cv2
import numpy as np

class Painter():
    def __init__(
        self,
        face_mesh_instance,
        display_width,
        display_height
    ):
        self.face_mesh = face_mesh_instance
        self.display_width = display_width
        self.display_height = display_height
        self.canvas = np.zeros((display_height, display_width, 1), np.uint8)
        self.last_gesture = ""
        self.cursor_position_buffer_for_opacity = []
        self.max_buffer_size = 100

    
    def paint_canvas(
        self,
        gesture,
        cursor_position
    ):
        self.canvas = np.zeros((self.display_height, self.display_width, 1), np.uint8)

        if not gesture['gesture'] == "None gesture recognized.":
            self.last_gesture = gesture['gesture']

        if gesture['gesture'] in ["left_click", "drag"]:
            self.__push_back_to_point_buffer("dot", cursor_position, 5, -1)
        elif gesture['gesture'] in ["right_click", "drop"]:
            self.__push_back_to_point_buffer("dot", cursor_position, 5, 2)
        elif gesture['gesture'] == "double_click":
            self.__push_back_to_point_buffer("dot", cursor_position, 10, -1)
        elif gesture['gesture'] == "scroll_up":
            self.__push_back_to_point_buffer("up", cursor_position)
        elif gesture['gesture'] == "scroll_down":
            self.__push_back_to_point_buffer("down", cursor_position)
        else:
            self.__push_back_to_point_buffer("none", None)
        
        for i, (command, point, radius, line_width) in enumerate(self.cursor_position_buffer_for_opacity):
            val = 2.55 * (i+1)
            if command is "none":
                continue
            elif command is "up":
                upper_point = (point[0], point[1]-50)
                cv2.arrowedLine(self.canvas, point, upper_point, (val, val, val), thickness=3)
            elif command is "down":
                under_point = (point[0], point[1]+50)
                cv2.arrowedLine(self.canvas, point, under_point, (val, val, val), thickness=3)
            else:
                cv2.circle(self.canvas, point, radius, (val, val, val), line_width)
        
        cv2.circle(self.canvas, cursor_position, 5, (150, 150, 150), -1)

        cv2.imshow('Canvas', self.canvas)


    def __push_back_to_point_buffer(
        self,
        command,
        cursor_position,
        radius=5,
        line_width=-1
    ):
        if self.__is_full():
            _ = self.__pop_first()
        self.cursor_position_buffer_for_opacity.append((
            command,
            cursor_position,
            radius,
            line_width
        ))


    def __is_full(self):
        return len(self.cursor_position_buffer_for_opacity) > self.max_buffer_size


    def __pop_first(self):
        cursor_position = self.cursor_position_buffer_for_opacity.pop(0)

        return cursor_position


    def paint_detail(
        self,
        face_gesture_processor,
        camera_frame
    ):
        self.face_mesh.paint_last_landmarks_to_image(camera_frame)
        
        camera_frame = cv2.flip(camera_frame, 1)

        last_face_info, _ = face_gesture_processor.buffer_for_face_info.face_info_buffer[-1]
        cv2.putText(camera_frame, f"left_eye_state: {last_face_info.left_eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(camera_frame, f"right_eye_state: {last_face_info.right_eye_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(camera_frame, f"rolling_angle: {last_face_info.rolling_angle:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(camera_frame, f"tilting_angle: {last_face_info.tilting_angle:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(camera_frame, f"panning_angle: {last_face_info.panning_angle:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(camera_frame, f"gesture_result: {self.last_gesture}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 3, cv2.LINE_AA)
        
        cv2.imshow('Camera frame', camera_frame)