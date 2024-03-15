# 实现对参考线的离散化操作，并计算每一段的弧长
'''
    最终的输出离散点，每个点应该包括
    - x 
    - y
    - kappa
    - s
    - heading angle
    即形如： [x,y,kappa,s,heading_angle]
'''
from scipy.interpolate import CubicSpline
from scipy.integrate import quad # 用于计算积分
from scipy.optimize import minimize_scalar
import numpy as np

import matplotlib.pyplot as plt
import ctypes

class referenceLineDiscretization:
    def __init__(self) -> None:
        self.TARGET_DISTANCE = 0.1 # 定义离散间距：米
        self.FIXED_POINT = [0,0] 
        self.NUM_POINTS = 0
        self.arc_list = []
        self.theta_ref_list = []
        self.kappa_ref_list = []
        self.discrete_points_x = []
        self.discrete_points_y = []
        
        
    def arc_length(self,cubic_spline,end_point_x):
        cs_derivative = cubic_spline.derivative()
        start_point_x = 0
        # fix me! 暂时只对开放曲线求积分，计算弧长 : ds = (dx^2 + dy^2) ^ (1/2) = (1 + (dy/dx)' ^2 ) ^(1/2) dx
        result,_ = quad(lambda x: np.sqrt(1+cs_derivative(x)**2),start_point_x,end_point_x)
        return result 
     
    def getTotalLength(self,cubic_spline,waypoints_x)->float:
        # 输入为所有waypoints的横坐标数组
        end_point_x = waypoints_x[-1]
        result = self.arc_length(cubic_spline,end_point_x)
        print("- Total length of spline : ",result)
        return result
    
    def getPerArcLength(self,cubic_spline,discrete_points_x):
        end_point_x = discrete_points_x[-1]
        result = self.arc_length(cubic_spline,end_point_x)
        # print("- Per Arc Length : ",result)
        self.arc_list.append(result)
        return result
    
    def getPointsNum(self,total_length)->int:
        num_points = int(np.floor(total_length / self.TARGET_DISTANCE)) + 1
        print("- Num of Points : ",num_points)
        self.NUM_POINTS = num_points
        return num_points
    
    # fix me! 以哪里为原点建立全局坐标系呢？(0,0)
    def calTheta_ref(self,discrete_points_x,discrete_points_y):
        # 弧度制
        theta_ref = np.arctan(discrete_points_y[-1]/discrete_points_x[-1])
        self.theta_ref_list.append(theta_ref)
        return theta_ref
    
    # fix me! 暂时只能对x单调递增情况生效，因为用的是三次样条曲线算的曲率           
    def calKappa_ref(self,cubic_spline,discrete_points_x,discrete_points_y):
        # 计算离散点的一阶导和二阶导
        y_prime = cubic_spline(discrete_points_x[-1],1)
        y_double_prime = cubic_spline(discrete_points_x[-1],2)
        kappa_ref = (y_double_prime) / np.sqrt(1 + y_prime**2)**3
        self.kappa_ref_list.append(kappa_ref)
        return kappa_ref
               
    def discreteUniformly(self,cubic_spline,waypoints_x:np.array):
        total_length = self.getTotalLength(cubic_spline,waypoints_x)
        num_points = self.getPointsNum(total_length)
        # 等间距离散化,输入x所有waypoints的横坐标数组
        discrete_points_x = [waypoints_x[0]]  # 起点
        discrete_points_y = [cubic_spline(waypoints_x[0])]
        
        
        for i in range(1,num_points):
            last_x = discrete_points_x[-1] # 上一个点的横坐标值
            # 寻找下一个离散点
            calDistance = lambda x:np.abs(self.arc_length(cubic_spline,x) - i*self.TARGET_DISTANCE)
            result = minimize_scalar(calDistance,bounds=(last_x, waypoints_x[-1]),method='bounded')
            
            # 更新离散点列表
            discrete_points_x.append(result.x)
            discrete_points_y.append(cubic_spline(result.x))
            self.discrete_points_x = discrete_points_x
            self.discrete_points_y = discrete_points_y  
            
            # 得到每个离散点距参考线起始点的弧长
            self.getPerArcLength(cubic_spline,discrete_points_x)
            # 得到每个离散点的theta_ref
            self.calTheta_ref(discrete_points_x,discrete_points_y)
            # 得到每个离散点的kappa_ref
            self.calKappa_ref(cubic_spline,discrete_points_x,discrete_points_y)
            
        return discrete_points_x,discrete_points_y
    
    def draw(self,waypoints_x:np.array,cubic_spline):
        discrete_points_x,discrete_points_y = self.discreteUniformly(cubic_spline,waypoints_x)
        # print("discrete_points_x : ",discrete_points_x)
        # print("discrete_points_y : ",discrete_points_y)
        plt.figure(figsize=(10,6))
        # 绘制三次样条曲线
        x_origin = np.linspace(waypoints_x[0],waypoints_x[-1],400)
        y_origin = cubic_spline(x_origin)
        plt.plot(x_origin,y_origin,label="Cubic Spline",linewidth=2)
        
        # 绘制曲线上的离散点
        plt.plot(discrete_points_x,discrete_points_y,'ro',label="Discrete Points",markersize=5)
        plt.title('Cubic Spline with Equal Arc Length Discretization')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
        
# if __name__ == "__main__":
#     test = referenceLineDiscretization()
#     waypoints_x = np.array([0,1,2,3,4])
#     waypoints_y = np.array([0,1,4,1,0])
#     cubic_spline = CubicSpline(waypoints_x,waypoints_y)
#     disc_points = test.discreteUniformly(cubic_spline,waypoints_x)
#     # visual = test.draw(waypoints_x,cubic_spline) 
#     arc_list = test.arc_list
#     theta_ref_list = test.theta_ref_list
#     kappa_ref_list = test.kappa_ref_list
#     print(kappa_ref_list)
    
#     plt.figure(1)
#     # x = np.linspace(0.2*np.pi,10000)
#     plt.plot(kappa_ref_list)
#     plt.show()

    