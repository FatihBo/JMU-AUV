import cv2
import numpy as np
import math

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
#        self.angle_int = int(self.angle)
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
        self.original = img
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

# img = cv2.imread(r'D:\github\Underwater-robot-competition\AUV_owen\screenshot.png')

# IP_box= imgprocess_detect_box(img)


# IP_box.cv_show('result',IP_box.processed)
# IP_box.cv_show('inRange_rbg',IP_box.rgb_ranged)
        


        
