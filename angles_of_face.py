import numpy as np
from typing import Dict, Tuple
import logging
from logger import logger


C = np.float64(1e3)
T = np.float64(1.)


class Angles_of_face():
    def __init__(self, coordinates_of_single_face:Dict[str, Tuple[np.float64]]):
        self.__nose_tip_coordinate = np.array(coordinates_of_single_face["nose_tip"], dtype=np.float64)
        self.__left_eye_tip_coordinate = np.array(coordinates_of_single_face["left_eye_tip"], dtype=np.float64)
        self.__right_eye_tip_coordinate = np.array(coordinates_of_single_face["right_eye_tip"], dtype=np.float64)


    def _get_angle_of_rolling_direction(self):
        # TODO: 여기 수정중
        left_eye_tip_xy_point = np.array(self.__left_eye_tip_coordinate[:2])*C
        right_eye_tip_xy_point = np.array(self.__right_eye_tip_coordinate[:2])*C

        gradient = self.__get_gradient(right_eye_tip_xy_point, left_eye_tip_xy_point)
        angle = np.arctan(gradient) * 180/np.pi  # [-90, 90] degree
        
        return angle
        

    def _get_angle_of_tilting_direction(self):
        nose_tip_yz_point = self.__nose_tip_coordinate[1:]

        norm = self.__get_norm(nose_tip_yz_point, np.array([0., 0.]))
        cos_theta = nose_tip_yz_point[1]**2/norm
        angle = np.arccos(cos_theta) * 180/np.pi

        return angle

    
    def _get_angle_of_panning_direction(self):
        nose_tip_xz_point = self.__nose_tip_coordinate[::2]

        norm = self.__get_norm(nose_tip_xz_point, np.array([0., 0.]))
        cos_theta = nose_tip_xz_point[1]**2/norm
        angle = np.arccos(cos_theta) * 180/np.pi

        return angle
    
    
    def __get_norm(
        self,
        vec_1:np.array,
        vec_2:np.array
        )->np.float64:
        displacement_vec = vec_1 - vec_2
        norm = np.sqrt(np.power((displacement_vec[0]),2) + np.power(displacement_vec[1],2))
        
        return norm


    def __get_gradient(
        self,
        vec_1:np.array,
        vec_2:np.array
        )->np.float64:
        displacement_vec = vec_1 - vec_2   
        gradient = np.divide(displacement_vec[1], displacement_vec[0], dtype=np.float64)
        
        return gradient
