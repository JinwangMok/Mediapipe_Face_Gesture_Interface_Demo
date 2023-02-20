import time
import numpy as np
from face_info_per_frame import Face_info_per_frame
from typing import Dict


class Face_angle_stabilizer():
    def __init__(self, stabilize_time=3.):
        self.stabilize_time:np.float64 = np.float64(stabilize_time)

        self.accumulated_time:np.float64 = np.float64(0.)
        self.last_frame_time:np.float64 = np.float64(0.)
        self.__frame_count:int = 0
        self.__sum_of_angles:Dict[str, np.float64] = {
            "rolling" : np.float64(0.),
            "tilting" : np.float64(0.),
            "panning" : np.float64(0.)
        }
        self.mean_of_angles:Dict[str, np.float64] = {
            "rolling" : np.float64(0.),
            "tilting" : np.float64(0.),
            "panning" : np.float64(0.)
        }


    def initialize(self):
        # This will be used when gesture command is "interface enable".
        # Because in this project, we can enable and disable the interface.
        self.last_frame_time:np.float64 = np.float64(0.)
        self.accumulated_time:np.float64 = np.float64(0.)
        self.__sum_of_angles:Dict[str, np.float64] = {
            "rolling" : np.float64(0.),
            "tilting" : np.float64(0.),
            "panning" : np.float64(0.)
        }
        self.mean_of_angles:Dict[str, np.float64] = {
            "rolling" : np.float64(0.),
            "tilting" : np.float64(0.),
            "panning" : np.float64(0.)
        }


    def stablize_angles(
        self, 
        face_info:Face_info_per_frame,
        )->Face_info_per_frame:
        if self.__is_ready():
            face_info.rolling_angle -= self.mean_of_angles["rolling"]
            face_info.tilting_angle -= self.mean_of_angles["tilting"]
            face_info.panning_angle -= self.mean_of_angles["panning"]
            return face_info

        if self.__is_first_step():
            self.__frame_count += 1
            self.__sum_of_angles["rolling"] = face_info.rolling_angle
            self.__sum_of_angles["tilting"] = face_info.tilting_angle
            self.__sum_of_angles["panning"] = face_info.panning_angle
            face_info.rolling_angle = np.float64(0.)
            face_info.tilting_angle = np.float64(0.)
            face_info.panning_angle = np.float64(0.)
            self.last_frame_time = np.float64(time.time())
            return face_info
        
        self.__frame_count += 1

        self.__calculate_mean_rolling_angle(face_info.rolling_angle)
        self.__calculate_mean_tilting_angle(face_info.tilting_angle)
        self.__calculate_mean_panning_angle(face_info.panning_angle)
        
        face_info.rolling_angle -= self.mean_of_angles["rolling"]
        face_info.tilting_angle -= self.mean_of_angles["tilting"]
        face_info.panning_angle -= self.mean_of_angles["panning"]

        self.accumulated_time = np.float64(time.time()) - self.last_frame_time

        return face_info


    def __is_ready(self):
        return self.accumulated_time > self.stabilize_time
    
    
    def __is_first_step(self):
        return self.last_frame_time == np.float64(0.)
    

    def __calculate_mean_rolling_angle(
        self,
        rolling_angle:np.float64
        )->np.float64:
        self.__sum_of_angles["rolling"] += rolling_angle
        self.mean_of_angles["rolling"] = self.__sum_of_angles["rolling"] / self.__frame_count


    def __calculate_mean_tilting_angle(
        self,
        tilting_angle:np.float64
        )->np.float64:
        self.__sum_of_angles["tilting"] += tilting_angle
        self.mean_of_angles["tilting"] = self.__sum_of_angles["tilting"] / self.__frame_count

    
    def __calculate_mean_panning_angle(
        self,
        panning_angle:np.float64
        )->np.float64:
        self.__sum_of_angles["panning"] += panning_angle
        self.mean_of_angles["panning"] = self.__sum_of_angles["panning"] / self.__frame_count