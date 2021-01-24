#Cameo类，实现了视频流的截图、录屏、保存等操作。
#已完成： 在managers类中嵌入图像处理模块
#已完成： 衡量图像处理算法性能的fps帧率的显示 （存在bug）
#已完成： 同时显示最终处理结果和中间处理过程的图像
import cv2
from managers import WindowManager, CaptureManager
import numpy
import time
import os

class Cameo(object):
    
    def __init__(self,camera_state,video_file):  #0/1为本机摄像头，-1为输入视频文件
        self._camera = camera_state
        self._video_file = video_file
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        if(camera_state == -1):
            self._captureManager = CaptureManager(cv2.VideoCapture(video_file),self._windowManager,True,False,False)
        else:
            self._captureManager = CaptureManager(cv2.VideoCapture(self._camera), self._windowManager, True,False,False)
        


     
    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.shouldMirrorPreview =False #是否镜像
            frame = self._captureManager.enterFrame
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self,keycode):
        if keycode == 32:  #space  空格保存
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  #tab  tab录制视频
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screenshot.avi')
            else:   
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":

    MODE = -1 #0/1为本机摄像头，-1为输入视频文件
    video_st = 1 #第 video_st 个视频
    video_file = r'D:\github\Underwater-robot-competition\AUV_owen\dataset'
    mode = 'line' #调用巡线模式处理算法
    #mode = 'box' #调用检测框模式处理算法

    video_file_list = os.listdir(video_file)
    video_file = video_file + '\\' + video_file_list[video_st]
    print(video_file)

    #video_file = r'D:\github\Underwater-robot-competition\underwater_video\a.avi'

    Cam = Cameo(MODE,video_file)
    Cam._captureManager.shouldprocess = True   #开启图像处理
    Cam._captureManager.shouldMirrorPreview = False
    Cam._captureManager.shouldshowFPS = True   #开启帧率显示
    Cam._captureManager.resize_fx_fy = (0.5,0.5) #对视频进行放缩
    Cam._captureManager.process_mode = mode
    Cam.run()
