"""This file implement many common methods and constant
Yuxin HU
"""

import numpy as np
import warnings

D2R = np.pi / 180


class QuadrotorFlyError(Exception):
    """General exception of QuadrotorFly"""
    def __init__(self, error_info):
        super().__init__(self)
        self.errorInfo = error_info
        warnings.warn("QuadrotorFly Error:" + self.errorInfo, DeprecationWarning)

    def __str__(self):
        return "QuadrotorFly Error:" + self.errorInfo


def get_rotation_matrix(att):
    cos_att = np.cos(att)
    sin_att = np.sin(att)

    rotation_x = np.array([[1, 0, 0], [0, cos_att[0], -sin_att[0]], [0, sin_att[0], cos_att[0]]])
    rotation_y = np.array([[cos_att[1], 0, sin_att[1]], [0, 1, 0], [-sin_att[1], 0, cos_att[1]]])
    rotation_z = np.array([[cos_att[2], -sin_att[2], 0], [sin_att[2], cos_att[2], 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    return rotation_matrix


def get_rotation_inv_matrix(att):
    att = -att
    cos_att = np.cos(att)
    sin_att = np.sin(att)

    rotation_x = np.array([[1, 0, 0], [0, cos_att[0], -sin_att[0]], [0, sin_att[0], cos_att[0]]])
    rotation_y = np.array([[cos_att[1], 0, sin_att[1]], [0, 1, 0], [-sin_att[1], 0, cos_att[1]]])
    rotation_z = np.array([[cos_att[2], -sin_att[2], 0], [sin_att[2], cos_att[2], 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_x, np.dot(rotation_y, rotation_z))

    return rotation_matrix


if __name__ == '__main__':
    try:
        raise QuadrotorFlyError('Quadrotor Exception Test')
    except QuadrotorFlyError as e:
        print(e)