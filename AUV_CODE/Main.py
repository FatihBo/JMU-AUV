#!/usr/bin/env python
#-*- coding: utf-8 -*-
'主函数'

'''
title           :AUV
description     :AUV(框架版)
date            :2021.1.21
version         :1.0
platform        :jetson nano
python_version  :3.7
'''

__author__='Fatih'

import sys
import Camera_Capture
import Arg
import cv2
import Pretre

#  如果是调试模式，则会显示窗口供给观看视频
#  如果是AUV脱缆运行模式，请改成False
Debug_mode = True

#读取程序运行参数
para = Arg.dataread()


if __name__=='__main__':
    cap = Camera_Capture.Camera_Init(para.at[0,'usborpi'], para.at[0,'win_width'], para.at[0,'win_height'], para.at[0,'fr_rate'])      #初始化摄像头0是行索引，字符是列索引
    while True:
        frame = Camera_Capture.Camera_Capture(para.at[0,'usborpi'], cap)

        if Debug_mode:
            cv2.imshow("Frame",frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
        else:
            pass


