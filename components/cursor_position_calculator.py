import numpy as np
from typing import Tuple
import logging
from logger import logger

class Cursor_position_calculator():
    def __init__(
        self,
        display_width,
        display_height
    ):
        self.dx, self.dy = self.__get_initial_depth_parameter(display_width, display_height)
        self.x_center = np.int32(display_width/2)
        self.y_center = np.int32(display_height/2)


    def calculate_cursor_position_from_face_info(
        self,
        last_face_info
    )->Tuple[np.int32]:
        # TODO: If cursor position looks unstable, add kalman filtering.
        
        theta_x = self.__degree_to_radian(-last_face_info.panning_angle)
        theta_y = self.__degree_to_radian(last_face_info.tilting_angle)

        x = np.int32(np.tan(theta_x) * self.dx) + self.x_center
        y = np.int32(np.tan(theta_y) * self.dy) + self.y_center
        
        return (x, y)


    def __get_initial_depth_parameter(
        self,
        width,
        height,
        max_theta=4.,
        margin=100
    )->Tuple[np.float64]:
        dx = np.float64((width/2 + margin) / np.tan(self.__degree_to_radian(max_theta)))
        dy = np.float64((height/2 + margin) / np.tan(self.__degree_to_radian(max_theta)))

        return (dx, dy)
    
    def __degree_to_radian(
        self,
        theta:np.float64
    )->np.float64:
        return np.float64(theta * np.pi / 180.)