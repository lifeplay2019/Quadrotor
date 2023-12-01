"""
Yuxin HU
2023.11.26
"""

import numpy as np
# import QuadrotorFly.SensorBase as SensorBase
from SensorBase import SensorBase, SensorType
import CommonFunctions as Cf

"""
*****************************************************************************************/
"""

D2R = Cf.D2R
g = 9.8


class ImuPara(object):
    def __init__(self, gyro_zro_tolerance_init=5, gyro_zro_var=30, gyro_noise_sd=0.01, min_time_sample=0.01,
                 acc_zgo_tolerance=60, acc_zg_var_temp=1.5, acc_noise_sd=300, name='imu'
                 ):
        """
        zro is zero rate output, sd is spectral density, zgo is zero g output
        :param gyro_zro_tolerance_init: the zero-bias of gyro, \deg/s
        :param gyro_zro_var: the noise variation in normal temperature, \deg/s
        :param gyro_noise_sd: the rate noise spectral density, \deg/s/\sqrt(Hz)
        :param acc_zgo_tolerance: the zeros of acc, mg
        :param acc_zg_var_temp: the noise variation vs temperature (-40~85), mg/(\degC)
        :param acc_noise_sd: the rate noise spectral density, mg\sqrt(Hz)
        :param name: the name of sensor
        :param min_time_sample: min sample time
        """
        # transfer the unit for general define
        # transfer to \rad/s
        self.gyroZroToleranceInit = gyro_zro_tolerance_init * D2R
        self.gyroZroVar = gyro_zro_var * (D2R**2)
        self.gyroNoiseSd = gyro_noise_sd  # i do not understand it, in fact
        # transfer to m/(s^2)
        self.accZroToleranceInit = acc_zgo_tolerance / 1000 * g
        std_temp = 1 / 1000 * g * 60  # assumed the temperature is 20 (\degC) here
        self.accZroVar = acc_zg_var_temp * (std_temp**2)
        self.accNoiseSd = acc_noise_sd  # i do not understand it, in fact
        self.name = name
        self.minTs = min_time_sample


mpu6050 = ImuPara(5, 30, 0.01)


class SensorImu(SensorBase):

    def __init__(self, imu_para=mpu6050):
        """
        :param imu_para:
        """
        SensorBase.__init__(self)
        self.para = imu_para
        self.sensorType = SensorType.imu
        self.angularMea = np.zeros(3)
        self.gyroBias = (1 * np.random.random(3) - 0.5) * self.para.gyroZroToleranceInit
        self.accMea = np.zeros(3)
        self.accBias = (1 * np.random.random(3) - 0.5) * self.para.accZroToleranceInit

    def observe(self):
        """return the sensor data"""
        return self._isUpdated, np.hstack([self.accMea, self.angularMea])

    def update(self, real_state, ts):
        """Calculating the output data of sensor according to real state of vehicle,
            the difference between update and get_data is that this method will be called when system update,
            but the get_data is called when user need the sensor data.
            the real_state here should be a 12 degree vector,
            :param real_state:
            0       1       2       3       4       5
            p_x     p_y     p_z     v_x     v_y     v_z
            6       7       8       9       10      11      12       13      14
            roll    pitch   yaw     v_roll  v_pitch v_yaw   a_x     a_y     a_z
            :param ts: system tick now
        """

        # process the update period
        if (ts - self._lastTick) >= self.para.minTs:
            self._isUpdated = True
            self._lastTick = ts
        else:
            self._isUpdated = False

        if self._isUpdated:
            # gyro
            noise_gyro = (1 * np.random.random(3) - 0.5) * np.sqrt(self.para.gyroZroVar)
            self.angularMea = real_state[9:12] + noise_gyro + self.gyroBias

            # accelerator
            acc_world = real_state[12:15] * 0.2 + np.array([0, 0, -g])
            # acc_world = np.array([0, 0, -g])
            rot_matrix = Cf.get_rotation_inv_matrix(real_state[6:9])
            acc_body = np.dot(rot_matrix, acc_world)
            noise_acc = (1 * np.random.random(3) - 0.5) * np.sqrt(self.para.gyroZroVar)
            # print(real_state[12:15], acc_body)
            self.accMea = acc_body + noise_acc + self.accBias
        else:
            # keep old
            pass

        return self.observe()

    def reset(self, real_state):
        """reset the sensor"""
        self._lastTick = 0

    def get_name(self):
        """get the name of sensor, format: type:model-no"""
        return self.para.name


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 2
    if testFlag == 1:
        s1 = SensorImu()
        flag1, v1 = s1.update(np.random.random(12), 0.1)
        flag2, v2 = s1.update(np.random.random(12), 0.105)
        print(flag1, "val", v1, flag2, "val", v2)

    elif testFlag == 2:
        from QuadrotorFly import QuadrotorFlyModel as Qfm
        q1 = Qfm.QuadModel(Qfm.QuadParas(), Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                                                           init_att=np.array([15, -20, 5])))
        s1 = SensorImu()
        t = np.arange(0, 10, 0.01)
        ii_len = len(t)
        stateArr = np.zeros([ii_len, 12])
        meaArr = np.zeros([ii_len, 6])
        for ii in range(ii_len):
            state = q1.observe()
            action, oil = q1.get_controller_pid(state)
            q1.step(action)

            flag, meaArr[ii] = s1.update(np.hstack([state, q1.acc]), q1.ts)
            stateArr[ii] = state

        estArr = np.zeros([ii_len, 3])
        estArrAcc = np.zeros([ii_len, 3])
        for ii in range(ii_len):
            if ii > 0:
                angle_dot = (meaArr[ii, 3:6] - s1.gyroBias) * q1.uavPara.ts
                estArr[ii] = estArr[ii - 1] + angle_dot
                meaAccTemp = meaArr[ii, 0:3] - s1.accBias
                acc_sum1 = np.sqrt(np.square(meaAccTemp[1]) + np.square(meaAccTemp[2]))
                acc_sum2 = np.sqrt(np.square(meaAccTemp[0]) + np.square(meaAccTemp[2]))
                estArrAcc[ii, 0] = -np.arctan2(meaAccTemp[0], acc_sum1)
                estArrAcc[ii, 1] = np.arctan2(meaAccTemp[1], acc_sum2)

        import matplotlib.pyplot as plt
        # plt.figure(1)
        for ii in range(3):
            plt.subplot(3, 1, ii + 1)
            plt.plot(t, stateArr[:, 6 + ii] / D2R, '-b', label='real')
            plt.plot(t, estArr[:, ii] / D2R, '-g', label='gyro angle')
            plt.plot(t, estArrAcc[:, ii] / D2R, '-m', label='acc angle')
        plt.show()
        # plt.figure(2)
        # plt.plot(t, stateArr[:, 3:6], '-b', label='real')
        # plt.plot(t, meaArr[:, 0:3], '-g', label='measure')
        # plt.show()
        # plt.plot(t, flagArr * 100, '-r', label='update flag')
