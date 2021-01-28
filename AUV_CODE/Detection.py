import cv2
import numpy as np
import logging
from collections import deque

'识别模块'

BUFF_ = 64
pts = deque(maxlen=BUFF_)
#距离检测参数
FORCAL_= 600
KNOW_NWIDTH_ = 25.2

#引导线检测
def guide_line_detect(mask, area_th=5000, aspect_th=0.8):
    '''

    TODO：部分时候很靠近边框时，会检测到框
    :param img:
    :param area_th:
    :param aspect_th:
    :return:
    '''
    ASPECT_RATIO_MIN = 0.15  # 重要参数
    MAX_CONTOUR_NUM = 6  # 如果出现更多的轮廓，不进行处理。这是为了对抗白平衡

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 目前对自动白平衡的处理，太多轮廓则直接返回
    candidates = []
    candidates_y = []
    if len(contours) < MAX_CONTOUR_NUM:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_th:  # 关键参数
                (x1, y1), (w1, h1), angle1 = cv2.minAreaRect(cnt)
                minAreaRect_area = w1 * h1
                aspect_ratio = float(w1) / h1
                if aspect_ratio > 1:
                    aspect_ratio = 1.0 / aspect_ratio
                    angle1 = np.mod(angle1 + 90, 180)

                extent = float(area) / minAreaRect_area

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area

                (x2, y2), (MA, ma), angle2 = cv2.fitEllipse(cnt)
                if angle2 > 90:
                    angle2 -= 180

                logging.debug('area %f,aspect_ratio %f,extent %f,solidity %f,angle1 %f,angle2 %f' % (
                area, aspect_ratio, extent, solidity, angle1, angle2))

                if aspect_ratio > aspect_th or aspect_ratio < ASPECT_RATIO_MIN or extent < 0.7 or solidity < 0.7 or abs(
                                angle1 - angle2) > 30:
                    break

                # img2 = img.copy()
                # contour_info(img2,area,aspect_ratio,extent,solidity,angle1,angle2,((x2, y2), (MA, ma), angle2))
                # cv.drawContours(img2, [cnt], 0, (0, 255, 0), 3)
                # show_img(img2)

                candidates.append((x1, y1, angle2))  # 目前这个组合是比较好的。
                candidates_y.append(y1)

        nc = len(candidates)
        if nc == 0:
            return None
        elif nc == 1:
            return candidates[0]
        else:
            logging.debug('multiple')

            idx = np.argmax(np.array(candidates_y))
            return candidates[idx]


#球识别函数
def BallColor_detect(frame,ballLower,ballUpper):
    '''
    用颜色阈值法检测圆
    :param frame:
    :param ballLower: 圆形的下线BGR
    :param ballUpper: 圆形的下线BGR
    :return: frame , hierarchy轮廓的数量 , center圆心坐标
    '''
    ballLower = tuple(ballLower)                        #传过来的是一个pd.Series，将类型强制转为tuple
    ballUpper = tuple(ballUpper)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, ballLower, ballUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    img,outline,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    center = None
    # if len(outline) > 0:                                       只要有轮廓就行，几个无所谓，相当于放松限制
    if len(outline)>0 and hierarchy<2:
        c = max(outline, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 255, 255), -1)
        findball_flag = True

    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(BUFF_ / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

    return frame,hierarchy,center


def BallHough_detect(frame,ball_minradius,ball_maxradius,ball_mindist):
    '''
    用霍夫变化检测圆
    :param frame:
    :param ball_minradius: 圆最小半径
    :param ball_maxradius: 圆最大半径
    :param ball_mindist: 圆心最小距离
    :return:frame , circle:圆心坐标与半径
    '''
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    circle = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,minDist=ball_mindist,minRadius=ball_minradius,maxRadius=ball_maxradius)
    if circle is not None:
        circle = np.round(circle[0,:]).astype("int")
        for (c_x,c_y,r) in circle:
            cv2.circle(frame,(c_x-5,c_y-5),(c_x+5,c_y+5),(0,255,0),-1)

    return frame,circle

def distance_to_camera(width, FORCAL_, PERWIDTH_):
    '''
    检测目标距离
    TODO ： 这里检测框的距离是不太准确的，如果要用，还是要重新测定
    :param width:目标宽度
    :param forcal: 焦距
    :param perwidth: 预测宽度
    :return:
    '''
    return ((width * FORCAL_) * 0.3048) / (12 * PERWIDTH_)

#目标识别，用于识别框线
#thresh0:前置摄像头二值化图像 frame0:前置摄像头图像
#Rect_Tarnum:是否识别到了框  data:框中点(cX,cY),外接矩形数据，框距离吗，外接矩形四点坐标
def Rect_Target_recognition(mask,frame,rect_area):
    '''
    用于检测框，并且测处框的距离
    :param mask:二值图像
    :param frame:
    :param rect_area: 设定的框大小，如果太小就不认为是要检测的框
    :return:data矩形的中心坐标以及长宽，距离以及角点坐标
    '''
    global cX
    Rect_Detflag = False                #未识别
    data = None
    cnts2 = []
    _, cnts3, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # findContours寻找轮廓
    for cnt in cnts3:
        area = cv2.contourArea(cnt)
        if area > rect_area:
            cnts2.append(cnt)
    if not (cnts2 == []):
        for c_3 in cnts2:
            M = cv2.moments(c_3)  # 求图形的矩
            cX = int((M["m10"] + 1) / (M["m00"] + 1))
            cY = int((M["m01"] + 1) / (M["m00"] + 1))

    if not (cnts2 == []):
        c = max(cnts2, key=cv2.contourArea)
        marker = cv2.minAreaRect(c)  # 得到最小外接矩形（中心(x,y),（宽，高），选住角度）
        metre = distance_to_camera(KNOW_NWIDTH_, FORCAL_, marker[1][0] + 1)  # 距离摄像头距离
        box = cv2.boxPoints(marker)  # 获取最小外接矩形的四个顶点
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame, "%.2fm" % (metre), (frame.shape[1] - 200, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        Rect_Detflag = True
        data = [cX,cY,marker, metre, box]
        return Rect_Detflag, data, frame
    return Rect_Detflag,data,frame


