#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""'abstract class for sensors, define the general call interface'
Yuxin HU
First Version: 2022.10.10
Recent 2023.11.26
"""

import numpy as np
import enum
from enum import Enum
import abc

"""
************************************************************************************************/
"""


class SensorType(Enum):
    """Define the sensor types"""
    none = enum.auto()
    imu = enum.auto()
    compass = enum.auto()
    gps = enum.auto()


class SensorBase(object, metaclass=abc.ABCMeta):
    """Define the abstract sensor_base class"""
    sensorType = SensorType.none

    def __init__(self):
        super(SensorBase, self).__init__()
        # the update tick of last one
        self._lastTick = 0
        self._isUpdated = False

    @property
    def last_tick(self):
        """the update tick of last one"""
        return self._lastTick

    @property
    def is_updated(self):
        return self._isUpdated

    def observe(self):
        """return the sensor data"""
        pass

    def update(self, real_state, ts):
        """Calculating the output data of sensor according to real state of vehicle,
            the difference between update and get_data is that this method will be called when system update,
            but the get_data is called when user need the sensor data.
            :param real_state: real system state from vehicle
            :param ts: the system tick
        """
        pass

    def reset(self, real_state):
        """reset the sensor"""
        pass

    def get_name(self):
        """get the name of sensor, format: type:model-no"""
        pass
