import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import process




def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path =r'D:\github\Underwater-robot-competition\AUV_owen\dataset\error_img\Snipaste_2021-02-03_17-20-16.jpg'
img_path = r'D:\github\Underwater-robot-competition\AUV_owen\dataset\error_img\Snipaste_2021-02-03_16-33-50.jpg'
img = cv2.imread(img_path)


start = time.time()
IP_box= process.imgprocess_detect_adsorbate(img)
img = IP_box.inRange_hsv
print("用时:"+str(time.time() - start))

src = IP_box.processed
h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = cv2.merge([img,img,img])
result[0:h,w:2*w,:] = IP_box.processed

cv_show('result_processed',result)