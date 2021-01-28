import numpy as np
import pandas as pd
import Camera_Capture
import Pretre

'增加，修改参数模块'

"""
设定多个Serial，每个Serial是一个特定功能的可调试参数
最后将多个Serial组成一个DataFrame写入文件argument.csv中
list xxx_name与xxx_value一一对应
"""


'''
调试参数

'''







argument = pd.DataFrame\
    ({  'usborpi': True,                            #使用USB摄像头或树莓派摄像头，True:USB
        'win_width': pd.Series(640),                #分辨率
        'win_height': 480,
        'fr_rate' : 30,                             #帧率
        'margin' : 20,
        'ballLower' : pd.Series([29, 86, 6]),       #用颜色阈值法检测球的时候所要使用的阈值范围
        'ballUpper' : pd.Series([64, 255, 255]),
        'ball_minradius' : 10,                      #检测球设定的最小半径
        'ball_maxradius' : 150,                     #检测球设定的最大半径
        'ball_mindist' : 20,                         #两个球之间的圆心最小的距离
        'rect_area' : 15000
                       })




def dataread():
    '''
    用于读取argument.csv的数据
    :return: data
    '''
    data = pd.read_csv('argument.csv')
    return data

def datawrite():
    '''
    每次修改完数据后要重新写入argument.csv
    :return: void 
    '''
    argument.to_csv('argument.csv')

