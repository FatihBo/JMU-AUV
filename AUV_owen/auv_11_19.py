import cv2
import numpy
import serial

#串口通信接口
portx = 'COM3'
bps = 9600
timex = 0.01
#ser = serial.Serial(portx, bps, timeout=timex)

#通信协议（2018版）
def PID_controlAUV(od_r,output):
    global model,AUV_dx,AUV_dy,AUV_dtheta,dl,dr
    print(output)
    head_bit = [0xaa,0x55]  # 两个字节为包头
    length_bit = [0x03]      #数据长度
    follow_bit = [0x08]      #用来选择三种模式
    control_bit = [0x00]  # 控制字节有效值：0-255
    time_level_bit = [0x00]  # 高四位为推进器动作时间，低四位为推进器推力的级数

    print(od_r)

    if od_r=='ball_down':
        follow_bit = [0x08]
        control_bit = [1+output]
        time_level_bit = [0x34]

    if od_r=='ball_up':
        follow_bit = [0x08]
        control_bit = [222+output]
        if control_bit[0]>=255:
            control_bit = [255]
        time_level_bit = [0x33]

    if od_r == 'left_translation':                      #左平移
        follow_bit = [0x08]
        control_bit = [35+output]
        time_level_bit = [0x33]
        model = 'trans'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = -0.01*output/30, 0, 0, 0, 0       #提供给里程表参数待修改
    elif od_r == 'right_translation':                    #右平移
        follow_bit = [0x08]
        control_bit = [128+output]
        time_level_bit = [0x33]
        model = 'trans'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0.01 * output/30, 0, 0, 0, 0       #提供给里程表参数待修改

    if od_r == 'left':                 #左旋转
        follow_bit = [0x08]
        control_bit = [91+output]
        time_level_bit = [0x33]
        model = 'dtheta'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0.026, 0, 0      # 提供给里程表参数待修改
    if od_r == 'right':                 #右旋转
        follow_bit = [0x08]
        control_bit = [128+output]
        time_level_bit = [0x33]
        model = 'dtheta'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, -0.026, 0, 0      # 提供给里程表参数待修改

    if od_r == 'go':
        follow_bit = [0x08]
        control_bit = [0xff]
        time_level_bit = [0x94]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0.2/30, 0.2/30               # 提供给里程表参数待修改

    if od_r == 'down':
        follow_bit = [0x0c]
        control_bit = [0x40]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    if od_r == 'up':
        follow_bit = [0x0c]
        control_bit = [0x00]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    if od_r == 'UP':
        follow_bit = [0x0c]
        control_bit = [0x20]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改
    if od_r == 'DOWN':
        follow_bit = [0x0c]
        control_bit = [0x30]
        time_level_bit = [0x02]
        model = 'diff'
        AUV_dx, AUV_dy, AUV_dtheta, dl, dr = 0, 0, 0, 0, 0  # 提供给里程表参数待修改

    parameter = head_bit + length_bit + follow_bit + control_bit + time_level_bit
    msg = parameter
    msg = bytearray(msg)

    try:  # 发送串口指令 与单片机通信
        ser.write(msg)
    except Exception as e:
        print("--异常--:", e)

    return 0

if __name__ == "__main__":
    cap0 = cv2.VideoCapture(0)
    
    
    while True:    
        ret0,frame0 = cap0.read()
        cv2.imshow('frame0',frame0)


        key = cv2.waitKey(1) & 0xFF#按键判断并进行一定的调整
        #按'a''d''w''s'分别将选框左移，右移
        # ，上移，下移
        #按'q'键退出录像
        if key == ord('s'): #后退
            print('后退')
            PID_controlAUV('back',80)

        elif key == ord('w'): #前景
            print('前进')
            PID_controlAUV('go',80)

        elif key == ord('d'): #右转
            print('右转')
            PID_controlAUV('right',80)
        elif key == ord('a'): #左转
            print('左转')
            PID_controlAUV('left',80)

        if key == ord('q'): #退出
            print('退出')