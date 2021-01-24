from threshold_ui import Ui_Widget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage,QPixmap,QFont
from PyQt5 import QtWidgets
import sys
import cv2
import numpy as np

class Threshold_Value_Edit(QtWidgets.QWidget,Ui_Widget):
    def __init__(self): #UI初始化
        super(Threshold_Value_Edit,self).__init__()
        self.setupUi(self)
        self.setWindowTitle("阈值编辑器")
        #img_path = r'D:\github\Underwater-robot-competition\AUV_owen\threshold_value_edit\Snipaste_2020-11-19_16-00-22.png'
        img_path = r'D:\github\Underwater-robot-competition\AUV_owen\screenshot.png'
        ft=QFont()
        ft.setPointSize(12)

        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider_2.setMaximum(255)
        self.horizontalSlider_3.setMaximum(255)
        self.horizontalSlider_4.setMaximum(255)
        self.horizontalSlider_5.setMaximum(255)
        self.horizontalSlider_6.setMaximum(255)

        self.horizontalSlider.setValue(255)
        self.horizontalSlider_3.setValue(255)
        self.horizontalSlider_5.setValue(255)
        self.label.setText("H最大值:"+str(self.horizontalSlider.value()))
        self.label_2.setText("H最小值:"+str(self.horizontalSlider_2.value()))
        self.label_3.setText("S最大值:"+str(self.horizontalSlider_3.value()))
        self.label_4.setText("S最小值:"+str(self.horizontalSlider_4.value()))
        self.label_5.setText("V最大值:"+str(self.horizontalSlider_5.value()))
        self.label_6.setText("V最小值:"+str(self.horizontalSlider_6.value()))
        self.label.setFont(ft)
        self.label_2.setFont(ft)
        self.label_3.setFont(ft)
        self.label_4.setFont(ft)
        self.label_5.setFont(ft)
        self.label_6.setFont(ft)
        self.lineEdit.setText("BGRmin:  BGRmax:")
        self.lineEdit.setFont(ft)
       
        self.img = cv2.imread(img_path)
        self.label_image_show(self.img)

        self.horizontalSlider.valueChanged[int].connect(self.horizontalSlider1_changeValue)
        self.horizontalSlider_2.valueChanged[int].connect(self.horizontalSlider2_changeValue)
        self.horizontalSlider_3.valueChanged[int].connect(self.horizontalSlider3_changeValue)
        self.horizontalSlider_4.valueChanged[int].connect(self.horizontalSlider4_changeValue)
        self.horizontalSlider_5.valueChanged[int].connect(self.horizontalSlider5_changeValue)
        self.horizontalSlider_6.valueChanged[int].connect(self.horizontalSlider6_changeValue)

    def label_image_show(self,img):
        if(len(img.shape)==2):
            img = cv2.merge([img,img,img])

        height, width, bytesPerComponent = img.shape   #返回的是图像的行数，列数，色彩通道数
        bytesPerLine = 3 * width    #每行的字节数        
        #cv2.cvtColor(img, cv2.COLOR_BGR2hsv, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_hsv888)         
        pixmap = QPixmap.fromImage(QImg)
        self.label_7.setPixmap(pixmap)
        #print(img.shape[0],img.shape[1])
        self.label_7.update()

    def label_image_show(self,img):
        if(len(img.shape)==2):
            img = cv2.merge([img,img,img])

        height, width, bytesPerComponent = img.shape   #返回的是图像的行数，列数，色彩通道数
        bytesPerLine = 3 * width    #每行的字节数        
        #cv2.cvtColor(img, cv2.COLOR_BGR2hsv, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)         
        pixmap = QPixmap.fromImage(QImg)
        self.label_7.setPixmap(pixmap)
        #print(img.shape[0],img.shape[1])
        self.label_7.update()





    def horizontalSlider1_changeValue(self):  #更新画板和滑条前的label
        self.label.setText("H最大值:"+str(self.horizontalSlider.value()))
        self.change_lineedit_value()
        self.BGR_img_change()
    def horizontalSlider2_changeValue(self):
        self.label_2.setText("H最小值:"+str(self.horizontalSlider_2.value()))
        self.change_lineedit_value()
        self.BGR_img_change()
    def horizontalSlider3_changeValue(self):
        self.label_3.setText("S最大值:"+str(self.horizontalSlider_3.value()))
        self.change_lineedit_value()
        self.BGR_img_change()
    def horizontalSlider4_changeValue(self):
        self.label_4.setText("S最小值:"+str(self.horizontalSlider_4.value()))
        self.change_lineedit_value()
        self.BGR_img_change()
    def horizontalSlider5_changeValue(self):
        self.label_5.setText("V最大值:"+str(self.horizontalSlider_5.value()))
        self.change_lineedit_value()
        self.BGR_img_change()
    def horizontalSlider6_changeValue(self):
        self.label_6.setText("V最小值:"+str(self.horizontalSlider_6.value()))
        self.change_lineedit_value()
        self.BGR_img_change()


    def change_lineedit_value(self):
        self.lineEdit.setText("BGRmin:["+str(self.horizontalSlider_2.value())+","+str(self.horizontalSlider_4.value())+","+str(self.horizontalSlider_6.value())+"] BGRmax:["+str(self.horizontalSlider.value())+","+str(self.horizontalSlider_3.value())+","+str(int(self.horizontalSlider_5.value()))+"]")

    def BGR_img_change(self):
       
        hsv_min = np.array([self.horizontalSlider_2.value(),self.horizontalSlider_4.value(),self.horizontalSlider_6.value()])
        hsv_max = np.array([self.horizontalSlider.value(),self.horizontalSlider_3.value(),self.horizontalSlider_5.value()])
        hsv = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)
        # dialet_BGR = cv2.dilate(BGR,None,iterations=2)
        # erode_BGR = cv2.erode(dialet_BGR, None, iterations=8)                   # 腐蚀 细的变粗
        inRange_hsv = cv2.inRange(hsv,hsv_min,hsv_max)
        self.label_image_show(inRange_hsv)
        # cv2.imshow('inRange_hsv',inRange_hsv)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
    
        print("hsv_min:")
        print(hsv_min)
        print("hsv_max:")
        print(hsv_max)        
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Threshold_Value_Edit()
    w.__init__()
    w.show()
    sys.exit(app.exec_())

