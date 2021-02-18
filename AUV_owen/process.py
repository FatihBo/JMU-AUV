import cv2
import numpy as np
import math
from utils import paint_chinese_opencv

object_color = 'white'

# color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
#               'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
#               'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
#               }

color_dist = {'white': {'Lower': np.array([26, 14, 164]), 'Upper': np.array([253, 115, 208])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }


class rect_line(object):  #水下机器人巡线的标志矩形 对象
    def __init__(self,cnt,img_shape):
        self.rect = cv2.minAreaRect(cnt) #最小面积拟合最大面积的矩形
        self.box = np.int0(cv2.boxPoints(self.rect))
        self.width = round(self.rect[1][0],2)  #矩形宽度
        self.height = round(self.rect[1][1],2) #矩形 高度
        self.x = round(self.rect[0][0],2)  #矩形中心x坐标
        self.y = round(self.rect[0][1],2)  #矩形中心y坐标 
        self.bottom_left_point,self.bottom_right_point,self.bottom_line_xy = self.bottom_line_center() 
        self.angle = round(90-self.two_dimention_angle((self.x-self.bottom_line_xy[0],self.y-self.bottom_line_xy[1]),(img_shape[0]-self.bottom_line_xy[0],self.bottom_line_xy[1]-self.bottom_line_xy[1])),2)
       
        #以下为整型数据
        self.x_int = int(self.x)
        self.y_int = int(self.y)

        self.width_int = int(self.width)
        self.height_int = int(self.height)
        self.area = self.width_int * self.height_int #面积大小


    def bottom_line_center(self): #获取底边x中心坐标,返回底边的左端点,右端点,底边中点
        first_point = self.box[0] #第一个点
        for p in self.box: #遍历判断第一个点是否是右下方的点
            if(p[0] != first_point[0] and p[1] > self.y and p[0] > first_point[0] ): #若p的y值大于中心点,并且其x坐标大于第一个点,则first_point不是右下方的点
                right_point = self.box[3] 
                left_point = self.box[0] 
                break
            else:
                right_point = first_point
                left_point = self.box[1]
        
        return (int(left_point[0]),int(left_point[1])),(int(right_point[0]),int(right_point[1])),(int((right_point[0]+left_point[0])/2),int((right_point[1]+left_point[1])/2))
        
    def two_dimention_angle(self,x,y): #两向量的角度
        x = np.array(x)
        y = np.array(y)
        Lx=np.sqrt(x.dot(x))
        Ly=np.sqrt(y.dot(y))
        cos_angle=x.dot(y)/(Lx*Ly)
        angle=np.arccos(cos_angle)
        angle2=angle*360/2/np.pi

        return angle2



    def cosVector(self,x,y):  #求两向量成角的余弦值
        if(len(x)!=len(y)):
            print('error input,x and y is not in the same space')
            return;
        result1=0.0;
        result2=0.0;
        result3=0.0;
        for i in range(len(x)):
            result1+=x[i]*y[i]   #sum(X*Y)
            result2+=x[i]**2     #sum(X*X)
            result3+=y[i]**2     #sum(Y*Y)
        return round(result1/((result2*result3)**0.5),2)  #保留2位小数

    def quasi_Euclidean_distance(self,point1,point2):  #准欧式距离 point1=(i,j) point2=(h,k)  已对运算速度优化
        if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]) : 
            return int(abs(point1[0]-point2[0]) + 0.4142* abs(point1[1]-point2[1]))
        else:
            return int(0.4142 * abs(point1[0]-point2[0]) + abs(point1[1]-point2[1]))     


class imgprocess_follow_line(object):
    def __init__(self,img):
        self.original = cv2.flip(img,0)
        self.processed,self.hsv_ranged,self.rect_box = self.priorprocess(img)

    
    def BGR_equalizeHist(self,img): #直方图均衡
        B,G,R = cv2.split(img)
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        R = cv2.equalizeHist(R)

        result = cv2.merge([B,G,R])
        return result


    def cv_show(self,name,img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def BGR_equalizeHist(self,img): #直方图均衡
        B,G,R = cv2.split(img)
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        R = cv2.equalizeHist(R)

        result = cv2.merge([B,G,R])
        return result

    def hist_show(self,img): #直方图
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])

        plt.show()

    def BGR_clahe(self,img):  #对比度受限的直方图均衡
        b,g,r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)

        clahed = cv2.merge([b,g,r])
        return clahed

    def maxAreacnt(self,cnts):  #求最大矩形
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

    def draw_box_points(self,img,box): #在矩形的四个角画上点
        for i,p in enumerate(box):
            cv2.circle(img,(p[0],p[1]),radius = 4,color = (0,0,255),thickness=-1)
            #cv2.putText(img,str(i),(p[0],p[1]),cv2.FONT_HERSHEY_SIMPLEX,1,color = 4)
        return img

    def cosVector(self,x,y):  #求两向量成角的余弦值
        if(len(x)!=len(y)):
            print('error input,x and y is not in the same space')
            return;
        result1=0.0;
        result2=0.0;
        result3=0.0;
        for i in range(len(x)):
            result1+=x[i]*y[i]   #sum(X*Y)
            result2+=x[i]**2     #sum(X*X)
            result3+=y[i]**2     #sum(Y*Y)
        return round(result1/((result2*result3)**0.5),2)  #保留2位小数

    def quasi_Euclidean_distance(self,point1,point2):  #准欧式距离 point1=(i,j) point2=(h,k)  已对运算速度优化
        if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]) : 
            return int(abs(point1[0]-point2[0]) + 0.4142* abs(point1[1]-point2[1]))
        else:
            return int(0.4142 * abs(point1[0]-point2[0]) + abs(point1[1]-point2[1]))        



    def priorprocess(self,img):
        line1 =None
        #img = self.BGR_clahe(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
        #dialet_hsv = cv2.dilate(hsv,None,iterations=2)
        #erode_hsv = cv2.erode(hsv, None, iterations=8)                   # 腐蚀 细的变粗
        inRange_hsv = cv2.inRange(hsv, color_dist[object_color]['Lower'], color_dist[object_color]['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #cv2.drawContours(img,cnts,-1,color = (255,0,0),thickness = 4)        
        if cnts:
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt) #最大外接矩形
                if(w*h>100 and h>w and w > 30):
                    continue
                    #cv2.rectangle(img,(x+5,y+5),(x+w-5,y+h-5),(0,255,0),2)
                    #cv2.circle(img,(int(x+w/2),int(y+h/2)),radius = 8,color = (0,255,255),thickness = -1)

    
        flag,cnt =self.maxAreacnt(cnts) 
        if flag: #是否存在
            line1 = rect_line(cnt,(img.shape[0],img.shape[1])) #一个方块对象
            if(line1.area > 1000 and line1.height > 50): #面积大于3000
                cv2.circle(img,(line1.x_int,line1.y_int),radius=3,color = (0,255,255),thickness=-1) #中心
                #cv2.putText(img,"Angle:"+str(line1.angle)+" w:"+str(line1.width_int)+"h:"+str(line1.height_int),(line1.x_int+2,line1.y_int+2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color=(255,255,255),thickness=1) #角度
                self.draw_box_points(img,line1.box)
                cv2.circle(img,line1.bottom_line_xy,radius=5,color = (0,0,0),thickness=-1) #标出底边中点
                cv2.circle(img,line1.bottom_left_point,radius=5,color = (255,0,0),thickness=-1) #标出底边左端点
                cv2.circle(img,line1.bottom_right_point,radius=5,color = (0,255,0),thickness=-1) #标出底边左端点
                cv2.putText(img,"L",line1.bottom_left_point,cv2.FONT_HERSHEY_SIMPLEX,1,color = (255,0,0)) #L标识
                cv2.putText(img,"R",line1.bottom_right_point,cv2.FONT_HERSHEY_SIMPLEX,1,color = (0,255,0)) #R标识
                cv2.line(img,(line1.x_int,line1.y_int),line1.bottom_line_xy,color = (0,0,0),thickness=2)  #角的一条边~
                #cv2.line(img,line1.bottom_line_xy,(img.shape[1],line1.bottom_line_xy[1]),color = (0,0,0),thickness=2) #角的另一条边~
                cv2.putText(img,"Angle:"+str(line1.angle)+" w:"+str(line1.width_int)+"h:"+str(line1.height_int),(line1.bottom_line_xy[0],line1.bottom_line_xy[1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color=(255,255,255),thickness=1) #角度
                cv2.line(img,(line1.bottom_line_xy[0],1),line1.bottom_line_xy,color = (0,0,255),thickness=1)
                cv2.line(img,(line1.x_int,line1.y_int),(int(img.shape[1]/2),line1.y_int),color = (255,0,0),thickness=2)
                cv2.putText(img,"x:"+str(line1.x_int-img.shape[1]/2),(int((line1.x_int+img.shape[1]/2)/2-3),line1.y_int-3),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color = (255,255,255),thickness=1)



        return img,inRange_hsv,line1

class particle(object): #粒子
    def __init__(self,xy):
        self.x = xy[0]
        self.y = xy[1]
        self.density = 0 #密集度
        self.life=False

class imgprocess_detect_box(object): #寻框
    def __init__(self,img):
        self.original = img
        self.HSVmin=np.array([112,28,131])
        self.HSVmax=np.array([193,253,255])
        self.processed,self.rgb_ranged = self.priorprocess(img)
       

    def cv_show(self,name,img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    def quasi_Euc_distance(self,point1,point2):  #准欧式距离 point1=(i,j) point2=(h,k)  已对运算速度优化
        if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]) : 
            return int(abs(point1[0]-point2[0]) + 0.4142* abs(point1[1]-point2[1]))
        else:
            return int(0.4142 * abs(point1[0]-point2[0]) + abs(point1[1]-point2[1]))        
       


    def abs_distance(self,point1,point2): #棋盘距离
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def cnt_area(self,cnt):#面积排序key
        area = cv2.contourArea(cnt)
        return area

    def particle_density(self,particle): #密集度排序key
        density = particle.density
        return density

    def draw_box_points(self,img,box): #在矩形的四个角画上点
        for i,p in enumerate(box):
            cv2.circle(img,(p[0],p[1]),radius = 4,color = (0,0,255),thickness=-1)
            cv2.putText(img,str(i),(p[0],p[1]),cv2.FONT_HERSHEY_SIMPLEX,1,color = 4)
        return img

    def particle_density_region(self,particle,lines): #对单个粒子进行密集度计算
        for line in lines:
            line = line[0]
            if(self.quasi_Euc_distance((particle.x,particle.y),(line[0],line[1])) < 40 or self.quasi_Euc_distance((particle.x,particle.y),(line[2],line[3]))<40):
                particle.density+=1
        return particle

    def hough_detect_box(self,img,inRange_rgb):    #霍夫变换精确定位框
        lines = cv2.HoughLinesP(inRange_rgb,0.8,np.pi/180,100,minLineLength=100,maxLineGap=5) #霍夫变换检测直线
        particle_list = []
        particle_set = [] #按范围去重后的列表
        rect_points = []
        if np.any(lines):
            for line in lines:
                x1, y1, x2, y2 = line[0]
                #cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType = cv2.LINE_AA)
                # cv2.circle(img,(x1,y1),radius=5,color = (0,0,0),thickness=-1)
                # cv2.circle(img,(x2,y2),radius=5,color = (0,0,0),thickness=-1)
                #计算粒子密集度
                line = line[0]
                pt1 = particle((line[0],line[1])) #实例化一个粒子对象
                pt1 = self.particle_density_region(pt1,lines) #计算该粒子周围的密集度

                pt2 = particle((line[2],line[3])) #实例化一个粒子对象
                pt2 = self.particle_density_region(pt2,lines) #计算该粒子周围的密集度

                particle_list.append(pt1)
                particle_list.append(pt2)
                
            #按粒子密集度排序
            particle_list.sort(key = self.particle_density,reverse=True)
            
            print(len(particle_list))

            leni = len(particle_list) - 1
            lenj = len(particle_list) - 1
            #取出较密区域的点
            for i,pointi in enumerate(particle_list):  
                for j,pointj in enumerate(particle_list) :
                    if self.abs_distance((pointi.x,pointi.y),(pointj.x,pointj.y)) <=1  and i != j :
                        particle_list[j].life=True
 
        

            for i in range(leni):
                if(particle_list[i].life):
                    particle_set.append(particle_list[i]) #存入新的列表

            # particle_set = particle_list            
            
            for i,point in enumerate(particle_set): #求取四个最密区域的点
                if(len(rect_points)==1): #已有第一个点
                    if(self.abs_distance((point.x,point.y),(rect_points[0].x,rect_points[0].y)) <= 5 ): #若是当前点与第一个矩形点的距离小于阈值
                        rect_points[0].x = (point.x + rect_points[0].x)/2 #均值
                        rect_points[0].y = (point.y + rect_points[0].y)/2 #均值
                    elif(self.abs_distance((point.x,point.y),(rect_points[0].x,rect_points[0].y)) >= 100): #若是当前点与第一个矩形点的距离大于阈值
                        rect_points.append(point) #判定为矩形第二个端点
                
                if(len(rect_points)==2): #已有第二个点
                    if(self.abs_distance((point.x,point.y),(rect_points[1].x,rect_points[1].y)) <= 5 ): #若是当前点与第二个矩形点的距离小于阈值
                        rect_points[1].x = (point.x + rect_points[1].x)/2 #均值
                        rect_points[1].y = (point.y + rect_points[1].y)/2 #均值
                        continue
                    elif( (abs(point.x - rect_points[0].x) + abs(point.y - rect_points[0].y)) >= 200 and (abs(point.x - rect_points[1].x) + abs(point.y - rect_points[1].y)) >= 200): #若是当前点与第一、二个矩形点的距离大于阈值
                        rect_points.append(point) #判定为矩形第三个端点
                
                if(len(rect_points)==3): #已有第三个点
                    if(self.abs_distance((point.x,point.y),(rect_points[2].x,rect_points[2].y)) <= 5 ): #若是当前点与第二个矩形点的距离小于阈值
                        rect_points[2].x = (point.x + rect_points[2].x)/2 #均值
                        rect_points[2].y = (point.y + rect_points[2].y)/2 #均值
                        continue
                    elif( (abs(point.x - rect_points[0].x) + abs(point.y - rect_points[0].y)) >= 200 and (abs(point.x - rect_points[1].x) + abs(point.y - rect_points[1].y)) >= 200 and (abs(point.x - rect_points[2].x) + abs(point.y - rect_points[2].y)) >= 200 ): #若是当前点与第一、二个矩形点的距离大于阈值
                        rect_points.append(point) #判定为矩形第三个端点
                        break        

                
                if(len(rect_points)==0): #第一个点
                    rect_points.append(point)
                
            # for i in range(len(particle_set)-1):
            #     cv2.circle(img,(int(particle_set[i].x),int(particle_set[i].y)),radius = 10,color = (255,255,255),thickness= -1 ) #画点
            if(len(rect_points)==4):
                cv2.circle(img,(int(rect_points[0].x),int(rect_points[0].y)),radius = 10,color = (0,0,0),thickness= -1 ) #画点
                cv2.putText(img,'0',(int(rect_points[0].x),int(rect_points[0].y)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
                cv2.circle(img,(int(rect_points[1].x),int(rect_points[1].y)),radius = 10,color = (0,0,0),thickness= -1 ) #画点
                cv2.putText(img,'1',(int(rect_points[1].x),int(rect_points[1].y)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
                cv2.circle(img,(int(rect_points[2].x),int(rect_points[2].y)),radius = 10,color = (0,0,0),thickness= -1 ) #画点
                cv2.putText(img,'2',(int(rect_points[2].x),int(rect_points[2].y)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
                cv2.circle(img,(int(rect_points[3].x),int(rect_points[3].y)),radius = 10,color = (0,0,0),thickness= -1 ) #画点
                cv2.putText(img,'3',(int(rect_points[3].x),int(rect_points[3].y)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
                

            for i,p in enumerate(rect_points):
                print(str(i)+":"+str(p.density)+" x:"+str(p.x)+" y:"+str(p.y))

        return img,rect_points 



    def priorprocess(self,img):
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        inRange_rgb = cv2.inRange(hsv,self.HSVmin,self.HSVmax)
        inRange_rgb = cv2.dilate(inRange_rgb,None,iterations=2)
        #inRange_rgb = cv2.erode(inRange_rgb, None, iterations=2)                   # 腐蚀 细的变粗

        cnts,hierarchy = cv2.findContours(inRange_rgb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(len(cnts)>0):
            cnts.sort(key = self.cnt_area, reverse=True) #按轮廓大小降序
     
            rect_box = rect_line(cnts[0],(img.shape[0],img.shape[1]))
            #img,hough_box = self.hough_detect_box(img,inRange_rgb)
            
            if(rect_box.x>100 and rect_box.y > 100):
                img = self.draw_box_points(img,rect_box.box)
                cv2.circle(img,(rect_box.x_int,rect_box.y_int),radius = 10,color = (0,255,255),thickness=-1)
                # cv2.line(img,(int(x-width/2),int(y-height/2)),(int(x+width/2),int(y-height/2)),(0,0,0),2)
                # cv2.line(img,(int(x-width/2),int(y+height/2)),(int(x+width/2),int(y+height/2)),(0,0,0),2)
        
                cv2.putText(img,"L",rect_box.bottom_left_point,cv2.FONT_HERSHEY_SIMPLEX,1,color = (255,0,0)) #L标识
                cv2.putText(img,"R",rect_box.bottom_right_point,cv2.FONT_HERSHEY_SIMPLEX,1,color = (0,255,0)) #R标识
                cv2.putText(img,"Angle:"+str(rect_box.angle)+" w:"+str(rect_box.width_int)+"h:"+str(rect_box.height_int),(rect_box.x_int,rect_box.y_int-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color=(255,255,255),thickness=1) #角度
            

        return img,inRange_rgb


class adsorbate_object: #吸附物对象,用于管道滤波中追踪与判定
    def _init_(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.area = self.w * self.h #面积
        self.type = 0 # 1为正方形 2为圆形
        self.lifetime = 0         


class imgprocess_detect_adsorbate(object):
    def __init__(self,img):
        self.original = img
        self.HSVmin = np.array([0,0,0])
        self.HSVmax = np.array([255,255,112])
        self.hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        self.inRange_hsv = cv2.inRange(self.hsv,self.HSVmin,self.HSVmax)
        #self.inRange_hsv = cv2.erode(self.inRange_hsv,None,iterations=8) #腐蚀，细的变粗
        self.inRange_hsv = cv2.dilate(self.inRange_hsv,None,iterations=2)
        self.RGB_thre = self.inRange_hsv 
        self.detect_data = np.array([0,0,0,0,0])
        self.processed = self.find_adsorbate(self.original)
    def find_adsorbate(self,img):
        cnts = cv2.findContours(self.inRange_hsv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.waitKey(40)
        object_list = []
        objective = 0
        ob_x=0
        ob_y=0
        ob_area=0
        detect_result= 0
        if cnts:
            #for cnt in cnts:
            cnts = sorted(cnts,key = cv2.contourArea,reverse = True)
            cnt = cnts[0]
            x,y,w,h = cv2.boundingRect(cnt)
            
            if(w*h>3000 and w*h < 50000):
                if(w-h < 1000):
                    #cv2.rectangle(img,(x+5,y+5),(x+w-5,y+h-5),(0,0,255),3)
                    cv2.putText(img,'area:'+str(w*h),(5,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0))
                    #print('area:'+str(w*h))
                    object_list.append([x,y,w,h]) #符合条件的存入列表
                    
                    #凸包点法识别方形/圆形            
                    hull = cv2.convexHull(cnt,returnPoints = True)
                    cv2.putText(img,str(hull.shape[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)

                    Hough_ROI = self.inRange_hsv[y:y+int(h),x:x+int(w)] 
                    circles = cv2.HoughCircles(Hough_ROI,cv2.HOUGH_GRADIENT,1,70,param1 = 10,param2=13.5,minRadius=0,maxRadius=1000)
                    if circles is not None: #若存在
                        if(hull.shape[0]>31):
                            #cv2.putText(img,'CIRCLE!',(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                            cv2.circle(img,(x+int(w/2),y+int(h/2)),int(w/2-3),(0,0,255),5)
                            detect_result = 2 #圆为2
                            objective = 1
                            ob_x = x
                            ob_y = y
                            ob_area = w*h
                    else:
                        if(hull.shape[0]<25):
                            #cv2.putText(img,'SQUARE!',(x,y),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))
                            cv2.rectangle(img,(x+5,y+5),(x+w-5,y+h-5),(255,0,0),3)
                            detect_result = 1 #方为1
                            objective = 1
                            ob_x = x
                            ob_y = y
                            ob_area = w*h

                    # if detect_result:
                    #     print(hull.shape[0],detect_result)      
        
        self.detect_data = np.array([objective,detect_result,ob_x,ob_y,ob_area])
   
        
        return img


class pipe:
    def __init__(self):
        self.pipe_life = [] #是否存在，存在为1，不存在为0
        self.pipe_x = [] #x坐标
        self.pipe_y = [] #y坐标
        self.pipe_area = [] #面积
        self.pipe_type = [] #类型 1为方，2为圆
        self.detect_result = [] #判定结果
        self.pipe_len = 0
        self.pipe_ob_nums = 0
        self.pipe_Maxlength = 40
        self.thre = 2 #判定
        self.pass_fps = 9

    def list_find(self,i): #life数据为0时，寻找其前pass_fps帧，判定是否存在
        if i>=self.pass_fps:
            sub_life = self.pipe_life[i-8:i]
        else:
            return False
        
        if sub_life.count(1)>3: #若前pass_fps帧存在次数大于3次          
            return True
        else:
            return False

    def lifetime_judge(self): #生命周期判定，若超过thre次触发判定为圆/方，则为圆/方
        start = 0
        end = 0
        continue_time = 0
        for i,life in enumerate(self.pipe_life):
            if life == 0:  #如果没有，那么有两种可能，1：之前有过，没检测到 2：之前就根本没有
                if start!= 0: #如果已开始,在前寻找
                    if self.list_find(i): #往前pass_fps帧找到了
                        end = i
                        continue_time = end - start
                        if continue_time > self.thre: #触发判定
                            square_num = self.pipe_type[start:end].count(1) #方的标记数量
                            circle_num = self.pipe_type[start:end].count(2) #圆的标记数量   
                            if square_num > circle_num:
                                self.detect_result.append(1) #判定为方
                            else:
                                self.detect_result.append(2) #判定为圆
                                
                    else: #往前pass_fps帧，啥也没有,重置
                        start = 0
                        end = 0
                        continue_time = 0
                        self.detect_result.append(0)
            else: #有存在目标
                if start==0: #还未记录
                    start = i
                else:
                    end = i
                    continue_time = end - start
                    if continue_time > self.thre: #触发判定
                        square_num = self.pipe_type[start:end].count(1) #方的标记数量
                        circle_num = self.pipe_type[start:end].count(2) #圆的标记数量   
                        if square_num > circle_num:
                            self.detect_result.append(1)
                        else:
                            self.detect_result.append(2)
                    else:
                        self.detect_result.append(0)
        if len(self.detect_result)>0:
            return self.detect_result[-1] 
        else:
            return 0

    def reboot_judge(self): #重启检查
        if len(self.pipe_life) > self.pipe_Maxlength:
            del self.pipe_life[0]
            del self.pipe_type[0]
            del self.pipe_x[0]
            del self.pipe_y[0]
            del self.pipe_area[0]
        

        
