import numpy as np
from dataclasses import dataclass


@dataclass
class MEC:
    mec_id: str
    x: float
    y: float
    capability_data_rate: int = 221760
    __task: list = None

    def __init__(self, mec_id, x, y):
        self.mec_id = mec_id
        self.x = x
        self.y = y
        self.__task = []

    def get_location(self):
        temp = np.array([self.x, self.y])
        temp.shape = (1, 2)
        return temp

    def get_location(self):
        temp = np.array([self.x, self.y])
        temp.shape = (1, 2)
        return temp

    def get_resource_utilize(self):
        return len(self.get_task()) * 800 / self.capability_data_rate

    def set_task(self, param):
        self.__task = param

    def get_task(self):
        return self.__task


@dataclass
class MEC_list:
    mec_list: list[MEC]

    def __init__(self):
        self.mec_list = []

    def query_mec(self, ue_xy: np.ndarray, MAX_DISTANCE: float):
        rows = []
        for mec in self.mec_list:
            D_VtoS = ((ue_xy[0][0] - mec.x) ** 2 + (
                    ue_xy[0][1] - mec.y) ** 2) ** 0.5
            if D_VtoS < MAX_DISTANCE:
                rows.append({"mec_id": mec.mec_id, 'distance': D_VtoS})
                return rows
