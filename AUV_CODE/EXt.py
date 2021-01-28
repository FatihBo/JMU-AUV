
'扩展模块'

#检测框距离函数
def distance_to_camera(width, forcal, perwidth):  # 距离计算
    return ((width * forcal) * 0.3048) / (12 * perwidth)
