import time
import numpy as np
from typing import List, Tuple
from copy import deepcopy
import logging
from logger import logger
from face_info_per_frame import Face_info_per_frame


class Buffer_for_face_info():
    def __init__(
        self,
        max_size:int=100,
        ):
        self.face_info_buffer:List[Tuple] = []
        self.max_size = max_size
    

    def push_back(
        self,
        face_info:Face_info_per_frame
        ):
        if self.__is_full():
            _ = self.__pop_first()
        self.face_info_buffer.append((
            face_info,
            np.float64(time.time())
        ))


    def __is_full(self):
        return len(self.face_info_buffer) > self.max_size


    def __pop_first(self):
        face_info = self.face_info_buffer.pop(0)

        return face_info


    def get_copied_face_info_buffer(self):
        copied_face_info_buffer = deepcopy(self.face_info_buffer)
        
        return copied_face_info_buffer


    def __is_empty(self):
        return len(self.face_info_buffer) == 0
    

    def get_copied_last_face_info(self):
        copied_last_face_info = deepcopy(self.face_info_buffer[-1])
        return copied_last_face_info