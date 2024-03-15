# 对于初次进行规划的情况，给定一个起始点坐标，然后进行寻找匹配点的操作；对于之后的规划，可能需要进行上下帧拼接和寻找匹配点处理；
# 还应包括对匹配点并不刚好是曲线上离散点的情况做插值处理，以求得更精确的匹配点
'''
    TODO
    1、首先要有参考线，以及参考线上离散点的各种信息
    2、计算初始规划点在参考线上的匹配点（在参考线上距离轨迹点最近的那个离散点），可以用一个索引来表示是线上的哪一个点，同时还可以得到最小距离值
    -- 不过这里先用暴力遍历吧，遍历每个ref上的离散点与初始规划点计算距离，求最小值）
    -- 还涉及到index_start、index_end 和 index_min，分情况如果index_min = 0，即匹配点是参考线的起点；还有前后加一减一，可能是后续插值
    3、前后点是为了两个向量做内积，找到投影点findProjectionPoint
    4、一阶的线性插值来计算匹配点的各种坐标值
    * 角度的归一化需要理解透彻，总要回到[-pi,pi]区间
    小结：
'''

import numpy as np
import math 
from referenceLineDiscretization import referenceLineDiscretization
from scipy.interpolate import CubicSpline
'''
    可用信息（列表内均按顺序存储）：
    cubic_spline: 三次样条曲线
    referenceLineDiscretization.arc_list = [] : 离散点的弧长s
    referenceLineDiscretization.theta_ref_list = [] ：离散点的yaw
    referenceLineDiscretization.kappa_ref_list = []：离散点的曲率
    referenceLineDiscretization.NUM_POINTS : 离散点的数量
    # 在调用discrete函数后能得到下面这两个
    referenceLineDiscretization.discrete_points_x = [] : 离散点的x坐标
    referenceLineDiscretization.discrete_points_y = [] : 离散点的y坐标
'''

class findMatchPoint:
    # 寻找参考线上距离规划起始点距离最近的离散点,同时还要返回这个离散点的索引
    def __init__(self) -> None:
        self.INDEX_MIN = 0
        self.INDEX_START = 10000 # 随便取的，为了不让它初始化等于INDEX_MIN
        self.INDEX_END = 100000
        self.DISTANCE_MIN = 999999
        # 用于存储寻找的匹配点的信息，列表里每个元素代表点的不同信息
        self.PATH_POINT_INFO = []
        
        '''
            用于接收来自referenceLineDiscretization的一些可用信息
            具体数据流向：
            referenceLineDiscretization -> findNearestDiscretePoint -> findProjectionPoint -> find类成员变量
        '''
        self.NUM_POINTS = 0
        self.arc_list = []
        self.theta_ref_list = []
        self.kappa_ref_list = []
        self.discrete_points_x = []
        self.discrete_points_y = []
        
    # planning_start_point 在初次规划时我就自己定义了一个点坐标
    def calDistance(self,planning_start_point_x,planning_start_point_y,reference_point_x,reference_point_y):
        dx = reference_point_x - planning_start_point_x
        dy = reference_point_y - planning_start_point_y
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def findNearestDiscretePoint(self,planning_start_point:np.array,
                                 num_points:int,
                                 discrete_points_x:np.array,
                                 discrete_points_y:np.array,
                                 arc_list:np.array,
                                 theta_ref_list:np.array,
                                 kappa_ref_list:np.array
                                 ):
        # 首先转存一些从ref来的信息，方便之后用到
        self.NUM_POINTS = num_points
        self.arc_list = arc_list
        self.theta_ref_list = theta_ref_list
        self.kappa_ref_list = kappa_ref_list
        self.discrete_points_x = discrete_points_x
        self.discrete_points_y = discrete_points_y
        
        # 开始执行本函数具体功能
        planning_start_point_x = planning_start_point[0]
        planning_start_point_y = planning_start_point[1]
        
        for i in range(1,num_points):
            distance = self.calDistance(planning_start_point_x,planning_start_point_y,discrete_points_x[i],discrete_points_y[i])
            if distance < self.DISTANCE_MIN:
                self.DISTANCE_MIN = distance
                self.INDEX_MIN = i
        
        # 寻找该离散点的前一点和后一点（按索引大小）
        if self.INDEX_START == self.INDEX_MIN:
            index_start = 0
        else:
            index_start = self.INDEX_MIN - 1
        self.INDEX_START = index_start
        
        if (self.INDEX_END + 1) == num_points:
            index_end = num_points
        else:
            index_end = self.INDEX_MIN + 1
        self.INDEX_END = index_end
        # 此处目的在于如果找到的距离最近的点恰好是投影点，那就直接返回；否则需要进入寻找投影点的过程
        if index_start == index_end:
            
            return discrete_points_x[index_start],discrete_points_y[index_start]
              
        return self.findProjectionPoint(discrete_points_x[index_start],
                                   discrete_points_y[index_start],
                                   discrete_points_x[index_end],
                                   discrete_points_y[index_end],
                                   planning_start_point_x,
                                   planning_start_point_y)
    
    def findProjectionPoint(self,discrete_point_start_x,
                            discrete_point_start_y,
                            discrete_point_end_x,
                            discrete_point_end_y,
                            planning_start_point_x,
                            planning_start_point_y):
        # 计算得到参考线上距离规划起始点最近的离散点及该离散点的前后点之后，计算投影点
        '''
        具体涉及向量内积：index_start点和规划起始点构成的向量 在 index_start点和index_end点构成的向量 上的投影
        通过向量内积公式可以求得规划点和投影点之间的距离
        '''
        vector_start2planning_x = planning_start_point_x - discrete_point_start_x
        vector_start2planning_y = planning_start_point_y - discrete_point_start_y
        vector_start2end_x = discrete_point_end_x - discrete_point_start_x
        vector_start2end_y = discrete_point_end_y - discrete_point_start_y
        
        # 向量点积
        vector_dot = vector_start2planning_x*vector_start2end_x + vector_start2planning_y*vector_start2end_y 
        
        # index_start点和规划起始点构成的向量的模 
        # vector_start2planning_norm = math.sqrt(vector_start2planning_x**2 + vector_start2planning_y**2)
        
        # index_start点和index_end点构成的向量的模
        vector_start2end_norm = math.sqrt(vector_start2end_x**2 + vector_start2end_y**2)
        
        # index_start点与投影点之间的距离
        delta_s = vector_dot / vector_start2end_norm
        
        # 进行线性插值，计算该投影点的相关信息：坐标、yaw、曲率等 ， 需要注意输入还要包括index_start点的弧长s
        return self.InterpolateUsingLinearApproximation(discrete_point_start_x,
                                                   discrete_point_start_y,
                                                   discrete_point_end_x,
                                                   discrete_point_end_y,
                                                   delta_s,
                                                   self.arc_list,
                                                   self.theta_ref_list,
                                                   self.kappa_ref_list
                                                   )
        
    # 线性插值，计算该投影点的相关信息：坐标、yaw、曲率等
    def InterpolateUsingLinearApproximation(self,discrete_point_start_x,
                                            discrete_point_start_y,
                                            discrete_point_end_x,
                                            discrete_point_end_y,
                                            delta_s,
                                            arc_list,
                                            theta_ref_list,
                                            kappa_ref_list):
        # 计算线性插值权重
        index_start_point_s = arc_list[self.INDEX_START]
        index_end_points_s = arc_list[self.INDEX_START + 2]
        
        delta_s_start2end = index_end_points_s - index_start_point_s
        weight = delta_s / delta_s_start2end
        
        # 计算该投影点坐标
        projection_point_x = (1 - weight) * discrete_point_start_x + weight * discrete_point_end_x
        projection_point_y = (1 - weight) * discrete_point_start_y + weight * discrete_point_end_y
        
        # 计算该投影点的yaw,这里偷懒使用线性插值，一般用slerp方法会更好
        # projection_point_theta = (1 - weight)*theta_ref_list[self.INDEX_START] + weight*theta_ref_list[self.INDEX_START+2]   
        
        # 计算该投影点的yaw，使用slerp方法实现
        projection_point_theta = self.slerp(theta_ref_list[self.INDEX_START],
                                            arc_list[self.INDEX_START],
                                            theta_ref_list[self.INDEX_START+2],
                                            arc_list[self.INDEX_START+2],
                                            delta_s+index_start_point_s)
        
        
        # 计算该投影点曲率, fix me! 暂时还未添加 dkappa 、ddkappa
        projection_point_kappa = (1 - weight)*kappa_ref_list[self.INDEX_START] + weight*kappa_ref_list[self.INDEX_START+2]
        
        # 添加点的信息
        temp_info = [projection_point_x,projection_point_y,math.degrees(projection_point_theta),projection_point_kappa]
        self.PATH_POINT_INFO = temp_info
        return temp_info
    
    # 用于给投影点计算theta
    def slerp(self,theta_ref_start,s_start,theta_ref_end,s_end,delta_s):
        if abs(s_end - s_start) <= 1e-6:
            print("time difference is too small!")
            return self.normalizeAngle(theta_ref_start)
        
        theta_ref_start_normal = self.normalizeAngle(theta_ref_start)
        theta_ref_end_normal = self.normalizeAngle(theta_ref_end)
        delta_theta_normal = theta_ref_end_normal - theta_ref_start_normal
        if (delta_theta_normal > math.pi):
            delta_theta_normal -= 2.0*math.pi
        elif (delta_theta_normal < -math.pi):
            delta_theta_normal += 2.0*math.pi
        
        # 线性插值
        weight_theta = (delta_s - s_start) / (s_end - s_start)
        theta_ref = theta_ref_start_normal + weight_theta * delta_theta_normal
        return self.normalizeAngle(theta_ref)
        
    # 对所有涉及角度的计算都将角度归一化到[-pi,pi]区间，仿照apollo的写法
    def normalizeAngle(self,angle):
        a = math.fmod(angle + math.pi, 2.0*math.pi)
        if a < 0.0:
            a += 2.0*math.pi
        return a - math.pi
    
    
# if __name__ == '__main__':
#     # 测试
#     pass
