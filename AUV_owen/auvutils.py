import serial

FORCAL = 600                # 计算距离函数设定的摄像头焦距，为定值
Know_Width = 25.2

#计算摄像头到框的距离
def distance_to_camera(perwidth):  # 距离计算,width为固定值。
    return ((Know_Width * FORCAL) * 0.3048) / (12 * perwidth)




class Ser(): #串口通信类
    def __init__(self,portx='COM1',bps = 9600,timex=0.01):
        self.Serial(portx,bps,timeout=timex)


def PID_controlAUV(od_r):
    global model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr
    
    head_bit = [0xAA,0x55]  # 两个字节为包头
    length_bit = [0x10]      #第三字节 数据长度 16 字节
    control1_mode = [0x00]   #第四字节 0x01表示深度锁定，0x02表示人工控制
    contral2_mode = [0x00]   #第五字节 0x01表示方向锁定，0x02表示随水动
    analog_sticks_front_back = []  #第六字节 摇杆模拟量 前后状态(0-255)，摇杆居中为128->停止
    analog_sticks_left_right = []  #第七字节 摇杆模拟量 左右状态(0-255),摇杆居中为128->停止
    verticle_move_control = [0x00] #第八字节 机器人垂直运动: 0x01表示向上,0x02表示向下,0x00表示不动作
    rotate_control = [0x00] #第9字节 旋转控制，0x01表示左旋，0x02表示右旋，0x00表示不动作
    throttle_size = [0x00]  #第10字节 油门大小,4档位可调,LB加档,LT减档 可分别设置4个档位油门大小
    led_brightness = [0x00] #第11字节 灯的亮度控制，0x01表示变亮，0x02表示变暗，0x00表示不动作
    camera_control = [0x00] #第12字节0x01表示聚焦，0x02表示变放焦，0x11放大，0x12缩小，0x00不动作
    pan_tilt = [0x00] #第13字节 0x01表示向上，0x02表示向下，0x03表示归中，0x00表示不动作
    mech_arm = [0x00]  #第14字节0x01表示张开，0x02表示关闭，0x00表示不动作
    Rpi = [0x00] #第15字节树莓派控制位0x00
    aspirator = [0x00] #第16字节吸取器控制位

    if(od_r == 'go'): #前进
        analog_sticks_front_back = [0xc8] #速度200
    if(od_r == 'back'): #后退
        analog_sticks_front_back = [0x38] #速度56 
    if(od_r == 'left'): #左进
        analog_sticks_left_right = [0x38] #速度56  小于128为左，大于128为右
    if(od_r == 'right'): #右进
        analog_sticks_left_right = [0xc8] #速度 200 
    if(od_r == 'up'): #上升
        verticle_move_control=[0x01] #上升
    if(od_r == 'down'): #下降
        verticle_move_control=[0x02] #下降
    if(od_r == 'turn_left'): #左转
        rotate_control=[0x01]
    if(od_r == 'turn_right'): #右转
        rotate_control=[0x02]



    parameter = head_bit + length_bit + control1_mode + contral2_mode + analog_sticks_front_back + analog_sticks_left_right + verticle_move_control
    parameter += rotate_control + throttle_size + led_brightness + camera_control + pan_tilt + mech_arm + Rpi + aspirator
    check_sum = sum(parameter)
    check_sum = [check_sum & 255]
    msg = parameter + check_sum
    print(msg)
    msg = bytearray(msg)
    try:  # 发送串口指令 与单片机通信
        #ser.write(msg)
        print(msg)
    except Exception as e:
        print("--异常--:", e)



#PID控制类
class PID:
    """PID Controller
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
        """Clears PID computations and coefficients"""
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
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
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
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

PID_controlAUV('go')