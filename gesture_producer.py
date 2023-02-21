import numpy as np
import logging
from logger import logger
from typing import Dict, Tuple
from buffer_for_face_info import Buffer_for_face_info
from buffer_for_cursor_position import Buffer_for_cursor_position


class Gesture_producer():
    def __init__(
        self,
        initial_cursor_position:Tuple[np.int32]
        ):
        self.initial_cursor_position = initial_cursor_position

        self.__buffer_for_face_info:Buffer_for_face_info
        self.__buffer_for_cursor_position:Buffer_for_cursor_position

        self.__flag_for_interface_enable = False
        self.__flag_for_double_click = False
        self.__flag_for_drag = False
        self.__last_frame_time_when_click = np.float64(0.)

        self.threshold_for_rolling = np.float64(40.)
        self.threshold_for_min_eye_closing_time = np.float64(0.2)
        self.threshold_for_max_eye_closing_time = np.float64(3.)
        self.threshold_for_right_click_to_drag_eye_closing_time = np.float64(1.)
        self.threshold_for_max_click_interval_time = np.float64(2.)


    def produce_gesture(
        self,
        buffer_for_face_info,
        buffer_for_cursor_position
        )->Dict:
        self.__buffer_for_face_info = buffer_for_face_info
        self.__buffer_for_cursor_position = buffer_for_cursor_position

        cursor_position = self.__buffer_for_cursor_position.cursor_position_buffer[-1]
        gesture = self.__calculate_gesture_from_face_info_buffer()

        if gesture == "interface_disable":
            cursor_position = self.initial_cursor_position

        result = {
            "cursor_position": cursor_position,
            "gesture": gesture
        }

        return result


    def __calculate_gesture_from_face_info_buffer(self):
        if len(self.__buffer_for_face_info.face_info_buffer) < 2:
            return "Preparing to recognize gesture..."

        if self.__is_interface_disabled():
            if self.__is_interface_on_off():
                self.__flag_for_interface_enable = True
                return "interface_enable"
            else:
                return "interface_disable"

        if self.__is_left_side_rolling():
            return "scroll_up"
        if self.__is_right_side_rolling():
            return "scroll_down"
        
        if self.__is_click():
            if self.__is_double_click():
                self.__flag_for_double_click = False
                self.__last_frame_time_when_click = self.__buffer_for_face_info.face_info_buffer[-1][1]
                return "double_click"
            else:
                self.__flag_for_double_click = True
                self.__last_frame_time_when_click = self.__buffer_for_face_info.face_info_buffer[-1][1]
                return "left_click"

        if self.__is_right_click():
            return "right_click"
        
        if self.__is_drag():
            self.__flag_for_drag = True
            logger.info("draging")
            return "drag"
        
        if self.__is_drop():
            self.__flag_for_drag = False
            return "drop"

        if self.__is_interface_on_off():
                self.__flag_for_interface_enable = False
                return "interface_disable"
        
        return "None gesture recognized."
    

    def __is_left_side_rolling(self):
        last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-1]
        
        if last_face_info.rolling_angle > self.threshold_for_rolling:
            return True
        else:
            return False


    def __is_right_side_rolling(self):
        last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-1]
        
        if last_face_info.rolling_angle < -self.threshold_for_rolling:
            return True
        else:
            return False


    def __is_click(self):
        last_face_info, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_both_eyes_closed(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False

        face_info_buffer_inverse = self.__buffer_for_face_info.face_info_buffer[::-1]
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
        if not self.__flag_for_double_click:
            return False

        _, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        if last_frame_time - self.__last_frame_time_when_click > self.threshold_for_max_click_interval_time:
            return False
        
        return True


    def __is_right_click(self):
        last_face_info, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_single_eye_opened(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False
        
        face_info_buffer_inverse = self.__buffer_for_face_info.face_info_buffer[::-1]
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
    

    def __is_drag(self):
        last_face_info, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_single_eye_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_single_eye_opened(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False
        
        face_info_buffer_inverse = self.__buffer_for_face_info.face_info_buffer[::-1]
        accumulated_time = np.float64(0.)
        for face_info, frame_time in face_info_buffer_inverse[1:]:
            if self.__is_single_eye_opened(face_info.left_eye_state, face_info.right_eye_state):
                accumulated_time += last_frame_time - frame_time
                last_frame_time = frame_time
            else:
                break
        if accumulated_time > self.threshold_for_right_click_to_drag_eye_closing_time:
            return True

        return False
    
    
    def __is_drop(self):
        last_face_info, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-2]

        if self.__flag_for_drag == False:
            return False
        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_single_eye_opened(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False
        
        return True


    def __is_interface_on_off(self):
        last_face_info, last_frame_time = self.__buffer_for_face_info.face_info_buffer[-1]
        before_the_last_face_info, _ = self.__buffer_for_face_info.face_info_buffer[-2]

        if not self.__is_both_eyes_opened(last_face_info.left_eye_state, last_face_info.right_eye_state):
            return False
        if not self.__is_both_eyes_closed(before_the_last_face_info.left_eye_state, before_the_last_face_info.right_eye_state):
            return False

        face_info_buffer_inverse = self.__buffer_for_face_info.face_info_buffer[::-1]
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


    def __is_interface_disabled(self):
        if self.__flag_for_interface_enable:
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