import numpy as np
from typing import List, Tuple
from copy import deepcopy

import logging
from logger import logger


class Buffer_for_cursor_position():
    def __init__(
        self,
        max_size:int=100,
    ):
        self.cursor_position_buffer:List[Tuple] = []
        self.max_size = max_size
    

    def push_back(
        self,
        cursor_position:Tuple[np.int32]
    ):
        if self.__is_full():
            _ = self.__pop_first()
        self.cursor_position_buffer.append(cursor_position)


    def __is_full(self):
        return len(self.cursor_position_buffer) > self.max_size


    def __pop_first(self):
        cursor_position = self.cursor_position_buffer.pop(0)

        return cursor_position


    def get_copied_cursor_position_buffer(self):
        copied_cursor_position_buffer = deepcopy(self.cursor_position_buffer)
        
        return copied_cursor_position_buffer


    def __is_empty(self):
        return len(self.cursor_position_buffer) == 0
    

    def get_copied_last_cursor_position(self):
        copied_last_cursor_position = deepcopy(self.cursor_position_buffer[-1])
        return copied_last_cursor_position