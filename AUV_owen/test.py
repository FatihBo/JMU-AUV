import numpy as np




def two_dimention_angle(x,y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle=np.arccos(cos_angle)
    angle2=angle*360/2/np.pi

    return angle2


x=np.array([3,5])
y=np.array([4,2])

print(two_dimention_angle(x,y))