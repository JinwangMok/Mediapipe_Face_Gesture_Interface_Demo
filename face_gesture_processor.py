import cv2
import numpy as np
import logging
from logger import logger

from components.face_info_per_frame import Face_info_per_frame
from components.face_angle_stabilizer import Face_angle_stabilizer
from components.buffer_for_face_info import Buffer_for_face_info
from components.cursor_position_calculator import Cursor_position_calculator
from components.buffer_for_cursor_position import Buffer_for_cursor_position
from components.gesture_producer import Gesture_producer


class Face_gesture_processor():
    def __init__(
        self,
        face_mesh_instance,
        display_width,
        display_height,
        max_buffer_size=100,
        stabilize_time=3.
    ):
        self.face_mesh = face_mesh_instance
        self.face_angle_stabilizer = Face_angle_stabilizer(stabilize_time=stabilize_time)
        self.buffer_for_face_info = Buffer_for_face_info(max_size=max_buffer_size)
        self.cursor_position_calculator = Cursor_position_calculator(display_width=display_width, display_height=display_height)
        self.buffer_for_cursor_position = Buffer_for_cursor_position(max_size=max_buffer_size)
        self.gesture_producer = Gesture_producer(initial_cursor_position=(np.int32(display_width/2), np.int32(display_height/2)))
    
    
    def process(
        self,
        image,
        target_face_randmarks
    ):
        coordinates_of_multi_face = self.face_mesh.get_3D_coordinates_from_bgr_image(image, target_face_randmarks)
        if self.__is_empty(coordinates_of_multi_face):
            return (False, None)

        face_info_per_frame = Face_info_per_frame(coordinates_of_multi_face['0'])
        stabilized_face_info_per_frame = self.face_angle_stabilizer.stablize_angles(face_info_per_frame)
        self.buffer_for_face_info.push_back(stabilized_face_info_per_frame)
        
        cursor_position_per_frame = self.cursor_position_calculator.calculate_cursor_position_from_face_info(stabilized_face_info_per_frame)
        self.buffer_for_cursor_position.push_back(cursor_position_per_frame)

        gesture = self.gesture_producer.produce_gesture(self.buffer_for_face_info, self.buffer_for_cursor_position)
        cursor_position = gesture['cursor_position']

        if gesture['gesture'] == "interface_enable":
            self.face_angle_stabilizer.initialize()

        return (True, (gesture, cursor_position))
    
    
    def __is_empty(self, collection_type):
        return len(collection_type) == 0