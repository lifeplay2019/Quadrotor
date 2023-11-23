# This is the course 5804 project
# 2023.11.20
# Name: YUXIN Hu (Shane)
"""
This is the Quadrotor Module
version 1.16
the aim is to create the math model for the quadrotor

"""

import numpy as np
import enum
from enum import Enum
import MemoryStorage
import SensorImu
import SensorBase
import SensorGps
import SensorCompass

# Some  key constant that been needed
# Radius to degree
D2R = np.pi / 180
# state space modle dimension [phi theta psi p q r u v w x y z] R12
state_dim = 12
# control vector ad action dimension [ft taux tauy tauz]
action_dim = 4
state_bound = np.array([10, 10, 10, 5, 5, 5, 80 * D2R, 80 * D2R, 180 * D2R, 100 * D2R, 100 * D2R, 100 * D2R, ])
action_bound = np.array([1, 1, 1, 1])


def rk4(func, x0, action, h):
    """
    Runge Kutta 4 order function
    :param func: system dynamic
    :param x0: system state
    :param action: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, action)
    k2 = func(x0 + h * k1 / 2, action)
    k3 = func(x0 + h * k2 / 2, action)
    k4 = func(x0 + h * k3, action)
    # print('rk4 debug: ', k1, k2, k3, k4)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class StructureType(Enum):
    quad_x = enum.auto()
    quad_plus = enum.auto()


class QuadParas(object):
    """Define the parameters of quadrotor model

    """

    def __init__(self, g=9.81,
                 rotor_num=4,
                 tim_sample=0.01,
                 structure_type=StructureType.quad_plus,
                 uav_l=0.450,
                 uav_m=1.50,
                 uav_ixx=1.75e-2,
                 uav_iyy=1.75e-2,
                 uav_izz=3.18e-2,
                 rotor_ct=1.11e-5,
                 rotor_cm=1.49e-7,
                 rotor_cr=646, rotor_wb=166,
                 rotor_i=9.90e-5,
                 rotor_t=1.36e-2):
        """quadrotor parameters
        These parameters are able to be estimation in web
        (https://flyeval.com/) since I do not have a real uav.
        the only one that is real has already been damage by myself
        common parameters:
            g          : N/kg,              gravity
            rotor-num  : int,              number of rotors, e.g. 4, 6, 8...
            tim_sample : s,                sample time of system
            structure_type:                quad_x, quad_plus
        uav:
            uav_l      : m,                distance from center of mass to center of rotor
            uav_m      : kg,               the mass of quadrotor
            uav_ixx    : kg.m^2            central principal moments of inertia of UAV in x
            uav_iyy    : kg.m^2            central principal moments of inertia of UAV in y
            uav_izz    : kg.m^2            central principal moments of inertia of UAV in z
        rotor (assume that four rotors are the same):
            rotor_ct   : N/(rad/s)^2,      lump parameter thrust coefficient, which translate rate of rotor to thrust
            rotor_cm   : N.m/(rad/s)^2,    lump parameter torque coefficient, like ct, usd in yaw
            rotor_cr   : rad/s,            scale para which translate oil to rate of motor
            rotor_wb   : rad/s,            bias para which translate oil to rate of motor
            rotor_i    : kg.m^2,           inertia of moment of rotor(including motor and propeller)
            rotor_t    : s,                time para of dynamic response of motor
        """
        self.g = g
        self.numOfRotors = rotor_num
        self.ts = tim_sample
        self.structureType = structure_type
        self.uavL = uav_l
        self.uavM = uav_m
        self.uavInertia = np.array([uav_ixx, uav_iyy, uav_izz])
        self.rotorCt = rotor_ct
        self.rotorCm = rotor_cm
        self.rotorCr = rotor_cr
        self.rotorWb = rotor_wb
        self.rotorInertia = rotor_i
        self.rotorTimScale = 1 / rotor_t


class SimInitType(Enum):
    rand = enum.auto()
    fixed = enum.auto()


class ActuatorMode(Enum):
    simple = enum.auto()
    dynamic = enum.auto()
    disturbance = enum.auto()
    dynamic_voltage = enum.auto()
    disturbance_voltage = enum.auto()


class QuadSimOpt(object):
    """
    parameters for guiding the simulation process
    """

    def __init__(self,
                 init_mode=SimInitType.rand,
                 init_att=np.array([5, 5, 5]),
                 init_pos=np.array([1, 1, 1]),
                 max_position=10,
                 max_velocity=10,
                 max_attitude=100,
                 max_angular=200,
                 sysnoise_bound_pos=0,
                 sysnoise_bound_att: object = 0,
                 actuator_mode=ActuatorMode.simple,
                 enable_sensor_sys=False):
        """

        :param init_mode:
        :param init_att:
        :param init_pos:
        :param max_position: maximum position
        :param max_velocity: maximum velocity
        :param max_attitude: maximun attitude
        :param max_angular: maximum angular velocity
        :param sysnoise_bound_pos: system noise bounded position
        :param sysnoise_bound_att: system noise bounded attitude
        :param actuator_mode: the mode of the actuator
        :param enable_sensor_sys:
        """
        self.initMode = init_mode
        self.initAtt = init_att
        self.initPos = init_pos
        self.maxPosition = max_position
        self.maxVelocity = max_velocity
        self.maxAttitude = max_attitude
        self.maxAngular = max_angular
        self.sysNoisePos = sysnoise_bound_pos
        self.sysNoiseAtt = sysnoise_bound_att
        self.actuatorMode = actuator_mode
        self.enableSensorSys = enable_sensor_sys


class QuadActuator(object):
    """
    Dynamic of  actuator about the motor and propeller
    """

    def __init__(self, quad_para: QuadParas, mode: ActuatorMode):
        """Parameters maintain together
        :param quad_para:   parameters of quadrotor,
        :param mode:        'simple': without dynamic of motor; 'dynamic' :with dynamic.
        """
        self.para = quad_para
        self.motorPara_scale = self.para.rotorTimScale + self.para.rotorCr
        self.motorPara_bias = self.para.rotorTimScale + self.para.rotorWb
        self.mode = mode

        # states of actuator
        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])

        # rate of the rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def dynamic_actuator(self, rotor_rate, action):
        """
        dynamic of the motor
        :param rotor_rate: input of the system, u
        :param action: action function (output)
        :return:
        """
        rate_dot = self.motorPara_scale * action + self.motorPara_bias - self.para.rotorTimScale * rotor_rate
        return rate_dot

    def reset(self):
        """
        reset all the states
        :return:
        """
        self.outTorque = np.zeros([self.para.numOfRotors])
        self.outThrust = np.zeros([self.para.numOfRotors])
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def step(self, action: 'int > 0'):
        """

        :param action:
        :return:
        """
        action = np.clip(action, 0, 1)
        if self.mode == ActuatorMode.simple:
            self.rotorRate = self.para.rotorCr * action + self.para.rotorWb
        elif self.mode == ActuatorMode.dynamic:
            self.rotorRate = rk4(self.dynamic_actuator, self.rotorRate, action, self.para.ts)
        else:
            self.rotorRate = 0
        self.outThrust = self.para.rotorCt * np.square(self.rotorRate)
        self.outTorque = self.para.rotorCm * np.square(self.rotorRate)
        return self.outThrust, self.outTorque


class QuadModel(object):
    """
     Module Interface, main class to describe the dynamic of the quadrotor
    """

    def __init__(self, uav_para: QuadParas, sim_para: QuadSimOpt):
        """
        init of quadrotor
        :param uav_para:  parameter of the uav
        :param sim_para:  parameter of the simple--without the dynamic, "dynamic"--with the dynamic
        """
        self.uavPara = uav_para
        self.simPara = sim_para
        self.actuator = QuadActuator(self.uavPara, sim_para.actuatorMode)

        # state of the quadrotor initial condition
        # Along with the position, velocity, attitude, angular velocity
        # and the linear acceleration
        self.position = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.attitude = np.array([0, 0, 0])
        self.angular = np.array([0, 0, 0])
        self.accelerate = np.zeros(3)

        # time control
        self._ts = 0

        # initial the sensor
        """
        to be continue
        """

    @property
    def ts(self):
        """
        return the tick of the system
        :return:
        """
        return self._ts

    def generate_init_att(self):
        """
         use to generate the initial attitude
         of the quadrotor in simPara
         """
        angle = self.simPara.initAtt * D2R
        if self.simPara.initMode == SimInitType.rand:
            phi = (1 * np.random.random() - 0.5) * angle[0]
            theta = (1 * np.random.random() - 0.5) * angle[1]
            psi = (1 * np.random.random() - 0.5) * angle[2]
        else:
            phi = angle[0]
            theta = angle[1]
            psi = angle[2]
        return np.array([phi, theta, psi])

    def generate_init_pos(self):
        """
        use to generate the initial position of the
        quadrotor in simPara
        :return:
        """
        pos = self.simPara.initPos
        if self.simPara.initMode == SimInitType.rand:
            x = (1 * np.random.random() - 0.5) * pos[0]
            y = (1 * np.random.random() - 0.5) * pos[1]
            z = (1 * np.random.random() - 0.5) * pos[2]
        else:
            x = pos[0]
            y = pos[1]
            z = pos[2]
        return np.array([x, y, z])

    def reset_state(self, att='none', pos='none'):
        """
        first set time to 0, and then reset the state for the position and attitude
        :param att:
        :param pos:
        :return:
        """
        self._ts = 0
        self.actuator.reset()
        if isinstance(att, str):
            self.attitude = self.generate_init_att()
        else:
            self.attitude = att

        if isinstance(pos,str):
            self.position = self.generate_init_pos()
        else:
            self.position = pos

        self.velocity = np.array([0, 0, 0])
        self.angular = np.array([0, 0, 0])

        # if the sensor system reset
        """
        to be continue
        """

    def dynamic_basic(self, state, action):
        """
        This is the main part of the dynamic function, it may cause a lot
        of computation, so it need to be fast.
        the majority of the dynamic_basic
        :param state:
        for linear position:
        0       1       2
        p_x     p_y     p_z
        for linear velocity:
        3       4       5
        v_x     v_y     v_z
        for angular position:
        6       7       8
        roll    pitch    yaw
        for angular velocity:
        9       10      11
        v_roll  v_pitch v_yaw
        :param action: u1(sum of thrust) u2(torque for roll) u3(torque for pitch)
                        u4(torque for yaw)
        :return: derivatives of the state....
        """
        # variable used
        att_cos = np.cos(state[6: 9])
        att_sin = np.sin(state[6: 9])
        noise_pos = self.simPara.sysNoisePos * np.random.random()
        noise_att = self.simPara.sysNoiseAtt * np.random.random()

        # dot state
        dot_state = np.zeros([12])

        # dynamic of position
        dot_state[0: 3] = state[3: 6]
        # if you want you can calculate the whole rotation matrix,
        # in here we just care the last column (R_zyx(phi, theta, psi))
        # also this is a SO(3) group
        dot_state[3: 6] = action[0] / self.uavPara * np.array([
             att_cos[2] * att_sin[1] * att_cos[0] + att_sin[2] * att_sin[0],
             att_cos[2] * att_sin[1] * att_cos[0] - att_sin[2] * att_sin[0],
             att_cos[0] * att_cos[1]
        ]) - np.array([0, 0, self.uavPara.g]) + noise_pos

        # dynamic of the attitude
        dot_state[6: 9] = state[6: 9]
        # this part confine the Coriolis and Centrifugal effect
        # This part of the variable is highly depend on the uav that you choose
        # or has been provided by the manufacture
        # This equation should be the same as the torque for the yaw
        rotor_rate_sum = (
            self.actuator.rotorRate[3] + self.actuator.rotorRate[2]
            - self.actuator.rotorRate[1] + self.actuator.rotorRate[0]
        )

        para = self.uavPara
        dot_state[6: 9] = np.array([
            state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
            - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
            + para.uavL * action[1] / para.uavInertia[0],
            state[9] * state[11] * (para.uavInertia[2] - para.uavInertia[0]) / para.uavInertia[1]
            + para.rotorInertia / para.uavInertia[1] * state[9] * rotor_rate_sum
            + para.uavL * action[2] / para.uavInertia[1],
            state[9] * state[10] * (para.uavInertia[0] - para.uavInertia[1]) / para.uavInertia[2]
            + action[3] / para.uavInertia[2]
        ]) + noise_att

        """
        this part is use for the future edit
        may be edit in the future
        """









