import serial
import time



class SerialCommuncaiton:
    '''
    串口通信模块
    port : 通信端口
    bps : 波特率
    timex : 通信重连时间
    '''
    def __init__(self):
        self.portx = '/dev/ttyUSB0'
        self.bps = 115200
        self.timex = 0.01
        self.head = [0xaa,0x55]
        self.ohters = [0x00,0x00]
        self.start_stop = 0x01
    def Init(self):

       ser = serial.Serial(self.portx, self.bps, timeout=self.timex)
       time.sleep(0.5)
       ser.write(b'\xaa\x55\x7F\x7F\x7F\x7F\x7F\x01\x00\x00\x01')

    def dataprocess(self,depth,LF,LB,RF,RB):
        self.depth = depth + 127
        self.direct = [LF+127,LB+127,RF+127,RB+127]
        pass

    def pack(self):
        parameter = self.head + self.depth + self.direct +self.ohters + self.start_stop
        check_sum = sum(parameter)
        check_sum = [check_sum & 255]
        msg = parameter + check_sum
        return msg


    def send(self,depth,LF,LB,RF,RB):
        dataprcess(depth,LF,LB,RF,RB)
        msg = pack()
        msg = bytearray(msg)
        try:  # 发送串口指令 与单片机通信
            ser.write(msg)
        except Exception as e:
            print("--异常--:", e)