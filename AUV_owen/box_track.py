import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import process






def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(r'D:\github\Underwater-robot-competition\AUV_owen\screenshot.png')


start = time.time()
IP_box= process.imgprocess_detect_box(img)
img = IP_box.rgb_ranged
print("用时:"+str(time.time() - start))
cv_show('result_processed',IP_box.processed)