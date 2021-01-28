import numpy as np
import pandas as pd
import Camera_Capture
import Pretre
import Arg
import time



class Frame:
    def __init__(self):
        '''
        self.usborpi = True    选择USB摄像头或者树莓派摄像头，默认为USB摄像头
        self.win_width/win_height = [640,320] 图像分辨率
        self.fr_rate = 30 图像帧率，这里的图像分辨率只对树莓派摄像头有用
        '''
        para = Arg.dataread()
        self.usborpi = para.at[0,'usborpi']
        self.win_width = para.at[0,'win_width']
        self.win_height = para.at[0,'win_height']
        self.fr_rate = para.at[0,'fr_rate']
    def Init(self,usborpi,win_width,win_height,fr_rate):
        '''
        初始化摄像头获得
        :param usborpi:
        :param win_width:
        :param win_height:
        :param fr_rate:
        :return:
        '''
        cap = Camera_Capture.Camera_Init(usborpi,win_width,win_height,fr_rate)
        frame = Camera_Capture.Camera_Capture(usborpi, cap)
        return frame



class Image:
    '''
    采单帧 图像
    TODO 读取一帧的图像，我这里不知道有没有必要，暂时没写
    '''
    pass

#PID控制类
class PID:
    """PID
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """PID参数清零"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value):
        """计算PID的值
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        error = self.SetPoint - feedback_value

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """设置P值"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """设置I值"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """设置D值"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """设置缓冲值
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """
        设置采样率

        """
        self.sample_time = sample_time
