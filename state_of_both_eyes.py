import numpy as np
from typing import Dict, Tuple


class State_of_both_eyes():
    def __init__(self, coordinates_of_single_face:Dict[str, Tuple[np.float64]]):
        self.__left_eye_upper_coordinate = np.array(coordinates_of_single_face["left_eye_upper"], dtype=np.float64)
        self.__left_eye_under_coordinate = np.array(coordinates_of_single_face["left_eye_under"], dtype=np.float64)
        self.__right_eye_upper_coordinate = np.array(coordinates_of_single_face["right_eye_upper"], dtype=np.float64)
        self.__right_eye_under_coordinate = np.array(coordinates_of_single_face["right_eye_under"], dtype=np.float64)

    
    def _get_open_close_state_of_both_eyes(self)->Tuple:
        # open:1, close:0
        pass
    