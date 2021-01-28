import cv2
import numpy as np

'预处理模块'

#BGR
BOUNDARIES_ = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]





#分水岭图像分割函数
# 用于二值化图像，分割出所需要的内容
def get_fg_from_hue_watershed_saturation(img, margin):
    '''
    分水岭算法分割二值图像
    :param img:
    :param margin:
    :return: mask 二值图像
    '''
    mask, hue = get_fg_from_hue(img, margin)

    mask_bg = cv2.inRange(hue, 60, 90)
    mask_bg = cv2.bitwise_or(mask_bg, cv2.inRange(hue, 128, 129))

    markers = np.zeros(mask.shape, np.int32)
    markers[mask == 255] = 1
    markers[mask_bg == 255] = 2

    cv2.watershed(img, markers)
    mask[markers == 1] = 255


    return mask

#HSV处理函数
def get_fg_from_hue(img, margin):
    '''
    :param img:
    :param margin:
    :return:
    '''
    FRACTION_AS_BLANK = 0.003
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dark = hsv[..., 2] < 32
    hsv[..., 0][dark] = 128

    dark = hsv[..., 1] < 50
    hsv[..., 0][dark] = 128

    mask = cv2.inRange(hsv[..., 0], np.array((0)), np.array((margin)))
    mask2 = cv2.inRange(hsv[..., 0], np.array((180 - margin)), np.array((180)))

    mask = cv2.bitwise_or(mask, mask2)

    if cv2.countNonZero(mask) < mask.shape[0] * mask.shape[1] * FRACTION_AS_BLANK:
        mask.fill(0)

    return [mask, hsv[..., 0]]


#颜色检测函数
def ColorDetection(frame):
    '''
    只检测设定BGR范围内的颜色，这个boundaries是需要自己在本模块设定的
    参数表不提供设定
    :param frame
    :param boundaries: 一个列表分别对应下线BGR值与上限BGR值
    :return: 二值图像
    '''
    for (lower, upper) in BOUNDARIES_:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)
        mask = cv2.bitwise_and(frame, frame, mask=mask)

    return mask

def morphology(mask):
    '''
    做形态学处理，这里主要是滤除较远处的干扰。
    :param mask:
    :return:
    '''
    thresh = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学开运算，简单滤除离框较远的干扰
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return mask