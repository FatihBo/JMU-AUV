import cv2
import numpy as np


lab = np.zeros([1,2,3],dtype = np.uint8)

lab[0][0] = [20,2,-36]
lab[0][1] = [100,87,72]

BGR = cv2.cvtColor(lab,cv2.COLOR_Lab2BGR)
hsv = cv2.cvtColor(BGR,cv2.COLOR_BGR2HSV)

print(hsv)