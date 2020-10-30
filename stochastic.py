import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import sympy
import random

from mpl_toolkits.mplot3d import Axes3D


def f(x1,x2):
    return (10*x1**2+x2**2)/2

def fx1(x1):
    return 10*x1

def fx2(x2):
    return x2

alpha=0.1
x1=-10
x2=10
GD_X1=[x1]
GD_X2=[x2]
GD_Y=[f(x1,x2)]
f_change=f(x1,x2)
iter_num=0

while(f_change>1e-10 and iter_num<200):
    x1_t=x1-alpha*fx1(x1)
    x2_t=x2-alpha*fx2(x2)
    f_t=f(x1_t,x2_t)

    f_change=np.absolute(f_t-f(x1,x2))
    x1=x1_t
    x2=x2_t
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_t)
    iter_num=iter_num+1

min_v=f(x1,x2)
print("minimum point:x=(",x1,",",x2,")")
print("minimum value:",min_v)

plt.plot(GD_X1,GD_X2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
X1=np.arange(-10,10,0.1)
X2=np.arange(-10,10,0.1)
X1,X2=np.meshgrid(X1,X2)#网格的创建，这个是关键
Y=(10*X1**2+X2**2)/2
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap='rainbow')
ax.plot(GD_X1,GD_X2,GD_Y,'ko-')
plt.savefig('fig.png',bbox_inches='tight')














