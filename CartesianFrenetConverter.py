# 实现cartesian和frenet互相转化
'''
    需要  rs rx ry rtheta rkappa 
          x y theta v a 
'''
import numpy as np
from math import *
import matplotlib.pyplot as plt
# import referenceLineDiscretization
from referenceLineDiscretization import referenceLineDiscretization
from scipy.interpolate import CubicSpline

class Convert():
    # only xy -> sd
    def cartesian2frenet(self,rs,rx,ry,theta_ref,x,y):
        s_condition = np.zeros(1)
        d_condition = np.zeros(1)
        
        # delta x , delta y
        dx = x - rx
        dy = y - ry
        
        # cos(theta_ref) , sin(theta_ref)
        cos_theta_ref = cos(theta_ref)
        sin_theta_ref = sin(theta_ref)
        
        # 叉乘，用于后续求l的正负号？如果是值的话，这个刚好就是可以求值
        cross_rd_nd = cos_theta_ref * dy - sin_theta_ref * dx
        # copysign，返回第一个参数的值和第二个参数的符号
        # 此处的根号下(dx*dx + dy*dy) 也是求l(即d)的值的，然后要加上符号，符号来自类似于数学的sign函数
        d_condition[0] = copysign(sqrt(dx*dx + dy*dy),cross_rd_nd)
        s_condition[0] = rs
        
        return s_condition,d_condition

    def frenet2cartesian(self,rs,rx,ry,theta_ref,s_condition,d_condition):
        # fabs方法以浮点数形式返回数字的绝对值，如math.fabs(-10) 返回10.0
        if fabs(rs - s_condition[0]) >= 1.0e-6:
            print("The reference point s and s_condition[0] doesn't match!")
        cos_theta_ref = cos(theta_ref)
        sin_theta_ref = sin(theta_ref)
        
        x = rx - sin_theta_ref * d_condition[0]
        y = ry + cos_theta_ref * d_condition[0]
        
        return x,y
    
    
    
if __name__ == "__main__":
    test = referenceLineDiscretization()
    waypoints_x = np.array([0,1,2,3,4])
    waypoints_y = np.array([0,1,4,1,0])
    cubic_spline = CubicSpline(waypoints_x,waypoints_y)
    disc_points = test.discreteUniformly(cubic_spline,waypoints_x)
    # visual = test.draw(waypoints_x,cubic_spline) 
    arc_list = test.arc_list
    theta_ref_list = test.theta_ref_list
    kappa_ref_list = test.kappa_ref_list
    # print(kappa_ref_list)
    
    
    test_convert = Convert()
    for i in range(1,test.NUM_POINTS):
        # c2f = test_convert.global2frenet(test.arc_list[-1],disc_points[0][-1],disc_points[1][-1],test.theta_ref_list[-1],)
        f2c = test_convert.frenet2cartesian(test.arc_list[-1],disc_points[0][-1],disc_points[1][-1],test.theta_ref_list[-1])
        
        
    plt.figure(1)
    x = np.linspace(0.2*np.pi,10000)
    plt.plot(kappa_ref_list)
    plt.show()