import numpy as np
from typing import Dict, Tuple
from angles_of_face import Angles_of_face
from state_of_both_eyes import State_of_both_eyes

class Face_information(Angles_of_face, State_of_both_eyes):
    def __init__(self, coordinates_of_single_face:Dict[str, Tuple[np.float64]]):
        super().__init__(coordinates_of_single_face)

        self.rolling_angle = self._get_angle_of_rolling_direction()
        self.tilting_angle = self._get_angle_of_tilting_direction()
        self.panning_angle = self._get_angle_of_panning_direction()

        self.left_eye_state, self.right_eye_state = self._get_open_close_state_of_both_eyes()  # open:1, close:0