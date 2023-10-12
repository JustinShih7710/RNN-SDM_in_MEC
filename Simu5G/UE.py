import numpy as np
from dataclasses import dataclass

import pandas as pd


@dataclass
class UE:
    vehicle_id: str
    veh_route: pd.DataFrame
    x: float
    y: float
    __time: int
    __task_bit_rate: int
    __task_list: list
    __mec_id: str
    __mec_server: list
    __resource_utilize: list
    __work_finished: list

    def __init__(self, vehicle_id: str, veh_route: pd.DataFrame):
        self.vehicle_id = vehicle_id
        self.veh_route = veh_route
        self.x = veh_route.vehicle_x[0]
        self.y = veh_route.vehicle_y[0]
        self.__time = 0
        self.__task_bit_rate = 0
        self.__task_list = []
        self.__mec_id = ''
        self.__mec_server = []

    def update_location(self, time):
        self.__time = time
        if len(self.veh_route[self.veh_route['data_timestep'] == time]) != 0:
            self.x = self.veh_route[self.veh_route['data_timestep'] == time].vehicle_x.values[0]
            self.y = self.veh_route[self.veh_route['data_timestep'] == time].vehicle_y.values[0]
            return True
        return False

    def get_next_location(self):
        x = 0
        y = 0
        if len(self.veh_route[self.veh_route['data_timestep'] == (self.__time + 1)]) != 0:
            x = self.veh_route[self.veh_route['data_timestep'] == (self.__time + 1)].vehicle_x.values[0]
            y = self.veh_route[self.veh_route['data_timestep'] == (self.__time + 1)].vehicle_y.values[0]
        temp = np.array([x, y])
        temp.shape = (1, 2)
        return temp

    def get_location(self):
        temp = np.array([self.x, self.y])
        temp.shape = (1, 2)
        return temp

    def reset(self):
        self.__mec_id = ''
        self.__mec_server = []
        self.__task_bit_rate = 0

    def get_mec_id(self):
        return self.__mec_id

    def set_mec_id(self, mec_id):
        self.__mec_id = mec_id

    def clr_mec_server(self):
        self.__mec_server = []

    def get_task_bit_rate(self):
        return self.__task_bit_rate

    def get_mec_server(self):
        return self.__mec_server

    def set_task_bit_rate(self, new_value):
        self.__task_bit_rate = new_value

    def get_task_list(self):
        return self.__task_list

    def set_mec_server(self, param):
        self.__mec_server = param

    def set_task(self, tmp):
        self.__task_list = tmp

    def get_task_finished_rate(self):
        return self.get_task_list().count(True) / len(self.get_task_list())
