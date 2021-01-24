import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import process

ball_color = 'red'
# SVmin:[86,118,95] HSVmax:[255,255,255]

color_dist = {'red': {'Lower': np.array([0, 128, 58]), 'Upper': np.array([255, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }



def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def BGR_equalizeHist(img): #直方图均衡
    B,G,R = cv2.split(img)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)

    result = cv2.merge([B,G,R])
    return result

def hist_show(img): #直方图
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    plt.show()

def BGR_clahe(img):  #对比度受限的直方图均衡
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    clahed = cv2.merge([b,g,r])
    return clahed
def maxAreacnt(cnts):  #求最大矩形
    area = 0
    max_cnt = 0
    flag = 0
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if(w*h > area and w > 50):
            area = w*h
            max_cnt = cnt

            flag = 1

    return flag,max_cnt

def draw_box_points(img,box): #在矩形的四个角画上点
    for p in box:
        cv2.circle(img,(p[0],p[1]),radius = 4,color = (0,0,255),thickness=-1)
    return img

            

img = cv2.imread(r'D:\github\Underwater-robot-competition\AUV_owen\screenshot.png')

IP_line= process.imgprocess_follow_line(img)
img = IP_line.processed
line1 = IP_line.line1


cv_show('result',img)