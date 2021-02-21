import cv2  
import numpy
import process
import os
os.remove('AUV_owen\original_data\original_data.txt')
avi_file_path1 = r'D:\github\Underwater-robot-competition\AUV_owen\dataset\recut\20210115_161634_Trim.mp4'
avi_file_path2 = r'D:\github\Underwater-robot-competition\AUV_owen\dataset\recut\20210115_162231_Trim.mp4'
cap = cv2.VideoCapture(avi_file_path1)
cap.set(cv2.CAP_PROP_FPS,20)
fps = cap.get(5)
frame_count = cap.get(7)
FIFO = process.pipe()
frame_num = 0
while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame,dsize=None,fx=0.5,fy=0.5)
    Processed = process.imgprocess_detect_adsorbate(frame)
    frame = Processed.processed
    data = Processed.detect_data #检测数据
    #数据读入管道
    FIFO.pipe_life.append(data[0]) 
    FIFO.pipe_type.append(data[1])
    FIFO.pipe_x.append(data[2])
    FIFO.pipe_y.append(data[3])
    FIFO.pipe_area.append(data[4])
    FIFO.reboot_judge() #限制长度

    #管道中判定

    ob_type = FIFO.lifetime_judge()  
    if ob_type == 1:
        cv2.putText(frame,'Rectangle!',(100,300),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0))
        print(frame_num,'BOX!')
    elif ob_type == 2:
        cv2.putText(frame,'Circle!',(100,300),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255))
        print(frame_num,'CIRCLE!')
    else:
        ob_type=0
        
    
    
    #数据写入
    data = ','.join(repr(e) for e in data) #字符串处理
    #print(data)
    with open('AUV_owen\original_data\original_data.txt','a',encoding='utf-8') as f:
       f.write( data +' '+str(ob_type)+ '\n')

    frame_num+=1
    #显示
    cv2.imshow("capture",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break