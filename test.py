from scipy.interpolate import CubicSpline
from scipy.integrate import quad # 用于计算积分
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
import math 

from referenceLineDiscretization import referenceLineDiscretization
from findMatchPoint import findMatchPoint

print("------------------test start!------------------------")

test_ref = referenceLineDiscretization()
waypoints_x = np.array([0,1,2,3,4])
waypoints_y = np.array([0,1,4,1,0])
cubic_spline = CubicSpline(waypoints_x,waypoints_y)
disc_points = test_ref.discreteUniformly(cubic_spline,waypoints_x)
# visual = test.draw(waypoints_x,cubic_spline) 
arc_list = test_ref.arc_list
theta_ref_list = test_ref.theta_ref_list
kappa_ref_list = test_ref.kappa_ref_list
# print(kappa_ref_list)

# 定义一个规划起始点，坐标与参考线起点重合
planning_start_point = np.array([2,2])

# 获取寻找findNearestDiscretePoint函数的输入
test_find = findMatchPoint()
num_points = test_ref.NUM_POINTS
discrete_points_x = test_ref.discrete_points_x
discrete_points_y = test_ref.discrete_points_y

match_point = test_find.findNearestDiscretePoint(planning_start_point,
                                                 num_points,discrete_points_x,
                                                 discrete_points_y,arc_list,
                                                 theta_ref_list,
                                                 kappa_ref_list)

print("----------------------finished the match process!------------------")
print('- match point:',test_find.PATH_POINT_INFO)    
print("- arc_list :",test_ref.arc_list)
print('- length of num_points :',test_ref.NUM_POINTS)
print("- length of arc_list :",len(test_ref.arc_list))
# visualization    
plt.figure(1)
# x = np.linspace(0.2*np.pi,10000)
plt.plot(discrete_points_x,discrete_points_y,'o',label="reference points")
plt.plot(test_find.PATH_POINT_INFO[0],test_find.PATH_POINT_INFO[1],'r*',label="match point")
plt.plot(planning_start_point[0],planning_start_point[1],'go',label="planning start point")
plt.show()


