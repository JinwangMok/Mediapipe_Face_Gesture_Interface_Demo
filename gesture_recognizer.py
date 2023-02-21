import numpy as np
import logging
from logger import logger
from typing import Dict, Tuple
from buffer_for_face_info import Buffer_for_face_info


class Gesture_recognizer():
    def __init__(self, display_width, display_height):
        self.buffer_for_face_info:Buffer_for_face_info

        self.flag_for_interface_enable = False
        self.flag_for_double_click = False
        self.last_frame_time_when_click = np.float64(0.)

        self.dx, self.dy = self.__get_initial_depth_parameter(display_width, display_height)
        self.threshold_for_rolling = np.float64(40.)
        self.threshold_for_min_eye_closing_time = np.float64(0.2)
        self.threshold_for_max_eye_closing_time = np.float64(3.)
        self.threshold_for_max_click_interval_time = np.float64(2.)


    def recognize_gesture_from_face_info_buffer(self, buffer_for_face_info):
        self.buffer_for_face_info = buffer_for_face_info

        if len(self.buffer_for_face_info.face_info_buffer) < 2:
            return "Ready to recognize gesture."

        if self.__is_left_side_rolling():
            return "scroll_up"
        if self.__is_right_side_rolling():
            return "scroll_down"
        
        if self.__is_click():
            if self.__is_double_click():
                self.flag_for_double_click = False
                self.last_frame_time_when_click = self.buffer_for_face_info.face_info_buffer[-1][1]
                return "double_click"
            else:
                self.flag_for_double_click = True
                self.last_frame_time_when_click = self.buffer_for_face_info.face_info_buffer[-1][1]
                return "left_click"

        if self.__is_right_click():
            # 여기는 만약 제스처 인포로 통합할 경우면 포인터 이동이랑 함께 구현해야 함
            return "right_click"

        if self.__is_interface_on_off():
            if self.__is_interface_enable():
                self.flag_for_interface_enable = True
                return "interface_enable"
            else:
                self.flag_for_interface_enable = False
                return "interface_disable"
        
        return "None gesture recognized."
    

    def __is_left_side_rolling(self):
        last_face_info, _ = self.buffer_for_face_info.face_info_buffer[-1]
        
        if last_face_info.rolling_angle > self.threshold_for_rolling:
            return True
        else:
            return False


    def __is_right_side_rolling(self):
        last_face_info, _ = self.buffer_for_face_info.face_info_buffer[-1]
        
        if last_face_info.rolling_angle < -self.threshold_for_rolling:
            return True
        else:
            return False


    def __is_click(self):
        last_face_info, last_frame_time = self.buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_both_eyes_closed(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False

        face_info_buffer_inverse = self.buffer_for_face_info.face_info_buffer[::-1]
        accumulated_time = np.float64(0.)
        for face_info, frame_time in face_info_buffer_inverse[1:]:
            if self.__is_both_eyes_closed(face_info.left_eye_state, face_info.right_eye_state):
                accumulated_time += last_frame_time - frame_time
                last_frame_time = frame_time
            else:
                break

        if (accumulated_time > self.threshold_for_min_eye_closing_time) and (accumulated_time < self.threshold_for_max_eye_closing_time):
            return True

        return False


    def __is_double_click(self):
        if not self.flag_for_double_click:
            return False

        _, last_frame_time = self.buffer_for_face_info.face_info_buffer[-1]
        if last_frame_time - self.last_frame_time_when_click > self.threshold_for_max_click_interval_time:
            return False
        
        return True


    def __is_right_click(self):
        last_face_info, last_frame_time = self.buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_single_eye_opened(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False
        
        face_info_buffer_inverse = self.buffer_for_face_info.face_info_buffer[::-1]
        accumulated_time = np.float64(0.)
        for face_info, frame_time in face_info_buffer_inverse[1:]:
            if self.__is_single_eye_opened(face_info.left_eye_state, face_info.right_eye_state):
                accumulated_time += last_frame_time - frame_time
                last_frame_time = frame_time
            else:
                break

        if (accumulated_time > self.threshold_for_min_eye_closing_time) and (accumulated_time < self.threshold_for_max_eye_closing_time):
            return True

        return False
    

    def __is_interface_on_off(self):
        last_face_info, last_frame_time = self.buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_both_eyes_closed(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False

        face_info_buffer_inverse = self.buffer_for_face_info.face_info_buffer[::-1]
        accumulated_time = np.float64(0.)
        for face_info, frame_time in face_info_buffer_inverse[1:]:
            if self.__is_both_eyes_closed(face_info.left_eye_state, face_info.right_eye_state):
                accumulated_time += last_frame_time - frame_time
                last_frame_time = frame_time
            else:
                break

        if (accumulated_time > self.threshold_for_min_eye_closing_time) and (accumulated_time > self.threshold_for_max_eye_closing_time):
            return True

        return False


    def __is_interface_enable(self):
        if self.flag_for_interface_enable:
            return False
        
        return True


    def __is_single_eye_opened(
        self,
        left_eye_state,
        right_eye_state
        ):
        if left_eye_state != right_eye_state:
            return True
        
        return False


    def __is_both_eyes_opened(
        self,
        left_eye_state,
        right_eye_state
        ):
        if left_eye_state and right_eye_state:
            return True

        return False


    def __is_both_eyes_closed(
        self,
        left_eye_state,
        right_eye_state
        ):
        if left_eye_state or right_eye_state:
            return False

        return True


    def __get_initial_depth_parameter(
        self,
        width,
        height,
        max_theta=4.,
        margin=100)->Tuple[np.float64]:
        dx = np.float64((width + margin) / (2. * np.tan(max_theta)))
        dy = np.float64((height + margin) / (2. * np.tan(max_theta)))

        return (dx, dy)