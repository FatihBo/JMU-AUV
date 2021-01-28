import time
import cv2
#import picamera       如果要用树莓派摄像头就按照对应的包，这里默认用USB摄像头

'视频推流模块'



def PiCamera_Init(win_width,win_height,fr_rate):
    '''
    使用树莓派摄像头
    TODO : 树莓派摄像头这里还没有测试过
    :param:win_width 画面宽度
    :param:win_height 画面高度
    :param:fr_rate 帧率
    :return:image
    '''
    camera = PiCamera()
    camera.resolution = (win_width,win_height)
    camera.framerate = fr_rate
    #camera.brightness = 50                #相机亮度,50为白平衡

    rawCapture = PiRGBArray(camera, size=(win_width,win_height))
    time.sleep(0.1)


    return rawCapture





def USBCamera_Init(win_width,win_height):
    '''
    使用USB摄像头
    :param:win_width 画面宽度
    :param:win_height 画面高度
    :param:fr_date 帧率
    :return:ret(bools/if frame is read correctly,it will be true) , frame data
    '''
    cap = cv2.VideoCapture(0)   # 前置摄像头
    cap.set(3, win_width)
    cap.set(4, win_height)
    return cap

def Camera_Init(flag,win_width,win_height,fr_rate):
    '''
    :param flag:initialization which kind of camera
    :param win_width:
    :param win_height:
    :param fr_rate:
    :return: cap: camera capture data
    '''
    if flag:
        cap = USBCamera_Init(win_width,win_height)
    else:
        cap = PiCamera_Init(win_width,win_height,fr_rate)

    return cap


def Camera_Capture(flag,cap):
    if flag:
        ret,frame = cap.read()
    else:
        cap.capture(rawCapture, use_video_port=True)
        frame = rawCapture.array

    return frame


