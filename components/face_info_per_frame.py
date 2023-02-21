import numpy as np
from typing import Dict, Tuple
import logging
from logger import logger


class Face_info_per_frame():
    def __init__(
        self,
        coordinates_of_single_face:Dict[str, Tuple[np.float64]]
    ):
        self.__nose_tip_coordinate = np.array(coordinates_of_single_face["nose_tip"], dtype=np.float64)
        self.__left_eye_tip_coordinate = np.array(coordinates_of_single_face["left_eye_tip"], dtype=np.float64)
        self.__left_eye_upper_coordinate = np.array(coordinates_of_single_face["left_eye_upper"], dtype=np.float64)
        self.__left_eye_under_coordinate = np.array(coordinates_of_single_face["left_eye_under"], dtype=np.float64)
        self.__right_eye_tip_coordinate = np.array(coordinates_of_single_face["right_eye_tip"], dtype=np.float64)
        self.__right_eye_upper_coordinate = np.array(coordinates_of_single_face["right_eye_upper"], dtype=np.float64)
        self.__right_eye_under_coordinate = np.array(coordinates_of_single_face["right_eye_under"], dtype=np.float64)
        
        self.rolling_angle = self._get_angle_of_rolling_direction()
        self.tilting_angle = self._get_angle_of_tilting_direction()
        self.panning_angle = self._get_angle_of_panning_direction()

        self.left_eye_state, self.right_eye_state = self._get_open_close_state_of_both_eyes()  #open:1, close:0

    
    def _get_angle_of_rolling_direction(self)->np.float64:
        left_eye_tip_xy_point = np.array(self.__left_eye_tip_coordinate[:2])
        right_eye_tip_xy_point = np.array(self.__right_eye_tip_coordinate[:2])

        gradient = self.__get_gradient(right_eye_tip_xy_point, left_eye_tip_xy_point)
        angle = np.arctan(gradient) * 180/np.pi  #[-90, 90] degree
        
        return angle
        

    def _get_angle_of_tilting_direction(self)->np.float64:
        nose_tip_yz_point = self.__nose_tip_coordinate[1:]

        norm = self.__get_norm(nose_tip_yz_point, np.array([0., 0.]))
        cos_theta = nose_tip_yz_point[1]**2/norm
        angle = np.arccos(cos_theta) * 180/np.pi

        return angle

    
    def _get_angle_of_panning_direction(self)->np.float64:
        nose_tip_xz_point = self.__nose_tip_coordinate[::2]

        norm = self.__get_norm(nose_tip_xz_point, np.array([0., 0.]))
        cos_theta = nose_tip_xz_point[1]**2/norm
        angle = np.arccos(cos_theta) * 180/np.pi

        return angle


    def __get_gradient(
        self,
        vec_1:np.array,
        vec_2:np.array
    )->np.float64:
        displacement_vec = vec_1 - vec_2   
        gradient = np.divide(displacement_vec[1], displacement_vec[0], dtype=np.float64)
        
        return gradient


    def __get_norm(
        self,
        vec_1:np.array,
        vec_2:np.array
    )->np.float64:
        displacement_vec = vec_1 - vec_2
        norm = np.sqrt(np.power((displacement_vec[0]),2) + np.power(displacement_vec[1],2))
        
        return norm


    def _get_open_close_state_of_both_eyes(self)->Tuple[np.bool]:
        #open:True, close:False
        left_eye_upper_y, left_eye_under_y = self.__left_eye_upper_coordinate[1], self.__left_eye_under_coordinate[1]
        right_eye_upper_y, right_eye_under_y = self.__right_eye_upper_coordinate[1], self.__right_eye_under_coordinate[1]
        
        left_eye_state = np.bool(True)
        right_eye_state = np.bool(True)
        
        threshold = 1e-2
        if left_eye_under_y - left_eye_upper_y < threshold:
            left_eye_state = np.bool(False)
        if right_eye_under_y - right_eye_upper_y < threshold:
            right_eye_state = np.bool(False)
        
        return (left_eye_state, right_eye_state)