import numpy as np
import math
import random
from sklearn.linear_model import RANSACRegressor, LinearRegression

def Calculate_Theta(A:float, B:float, C:float = 1):
    if A >= 0:
        return math.acos(C/math.sqrt(A**2 + B**2 + C**2)) * 180 / math.pi
    else:
        return - math.acos(C/math.sqrt(A**2 + B**2 + C**2)) * 180 / math.pi

def Mask(height, width, inlier_mask):
    assert height * width == len(inlier_mask)
    mask = np.zeros((height, width))
    for i in range(len(inlier_mask)):
        mask[i // width, i % width] = 1 if inlier_mask[i] else 0
    return mask
    
def Plane_RANSAC(Points):
    height, width = Points.shape[0], Points.shape[1]
    Points = Points.reshape(-1, 3)
    xy = Points[:, 0:2]
    z = Points[:, 2].reshape(-1,)
    
    ransac = RANSACRegressor()
    ransac.fit(xy, z)
    ransac_A = ransac.estimator_.coef_[0]
    ransac_B = ransac.estimator_.coef_[1]
    ransac_D = ransac.estimator_.intercept_
    ransac_theta = Calculate_Theta(ransac_A, ransac_B, 1)
    inlier_mask = ransac.inlier_mask_
    mask = Mask(height, width, inlier_mask)
    # print(f"ransac coef:{ransac.estimator_.coef_}, ransac intercept:{ransac.estimator_.intercept_}")
    
   
    lr = LinearRegression()
    lr.fit(xy, z)
    # print(f"lr coef:{lr.coef_}, lr intercept:{lr.intercept_}")
    lr_A = lr.coef_[0]
    lr_B = lr.coef_[1]
    lr_D = lr.intercept_
    lr_theta = Calculate_Theta(lr_A, lr_B, 1)
    
    Res = np.array([ransac_A, ransac_B, ransac_D, ransac_theta, lr_A, lr_B, lr_D, lr_theta])
    
    return Res, mask

def Calculate_Plane(x1, x2, x3):
    """
    输入三点坐标，计算平面方程
    """
    v1 = np.array([x1[0]-x2[0], x1[1]-x2[1], x1[2]-x2[2]])
    v2 = np.array([x1[0]-x3[0], x1[1]-x3[1], x1[2]-x3[2]])
    k = np.cross(v1, v2)
    D = -np.dot(k, x1)
    return [k[0], k[1], k[2], D]
    
    
def Get_Mask(Points_shape, inliers_idx):
    mask = np.zeros((Points_shape[0], Points_shape[1]))
    for i in range(len(inliers_idx)):
        r = inliers_idx[i] // Points_shape[1]
        c = inliers_idx[i] % Points_shape[1]
        mask[r, c] = 1
    return mask
    
def Least_Square(Points):
    """
    最小二乘法计算平面方程
    """
    assert Points.shape[1] == 3    
    A = Points[:, 0:2]
    A = np.c_[A, np.ones(A.shape[0])]
    K, rest, _, sig = np.linalg.lstsq(A, Points[:, 2], rcond=-1)
    print(K, rest, sig)
    
"""
def Plane_RanSAC(Points):
    # 希望得到正确模型的概率
    P = 0.99
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    iters = 100000
    # 数据和模型之间可接受的差值
    sigma = 0.25
    # 期望得到内点的比例
    w = 0.5
    # 图像大小
    Points_shape = [Points.shape[0], Points.shape[1]]
    # Points reshape
    Points = Points.reshape(-1, 3)
    
    # 最好模型的参数估计和内点数目
    best_A = 0
    best_B = 0
    best_C = 0
    best_D = 0
    best_inliers_num = 0
    
    for i in range(iters):
        # 随机在数据中红选出三个点去求解模型
        sample_index = random.sample(range(Points.shape[0]),3)
        x1 = Points[sample_index[0]]
        x2 = Points[sample_index[1]]
        x3 = Points[sample_index[2]]
        # 计算平面方程
        A, B, C, D = Calculate_Plane(x1, x2, x3)
        # 计算内点数目
        inliers_num = 0
        # 内点索引
        inliers_idx = []
        # 最大 error
        max_error = 0
        for j in range(Points.shape[0]):
            if math.fabs(A*Points[j, 0] + B*Points[j, 1] + C*Points[j, 2] + D) > max_error:
                max_error = math.fabs(A*Points[j, 0] + B*Points[j, 1] + C*Points[j, 2] + D)
            if math.fabs(A*Points[j, 0] + B*Points[j, 1] + C*Points[j, 2] + D) < sigma:
                inliers_num += 1
                inliers_idx.append(j)
        print(max_error)
                
        # 判断当前的模型是否比之前估算的模型好
        if inliers_num > best_inliers_num:
            best_inliers_num = inliers_num
            best_A = A
            best_B = B
            best_C = C
            best_D = D
            # 计算得到当前模型的成功概率
            # e = 1 - inliers_num / Points.shape[0]
            # iters = math.log(1 - P) / math.log(1 - (1 - e) ** 3)
            iters = math.log(1 - P) / math.log(1 - pow(inliers_num / (Points.shape[0] * 2), 2))
    
        if best_inliers_num > w * Points.shape[0]:
            break
    
    mask = Get_Mask(Points_shape, inliers_idx)
        
    return best_A, best_B, best_C, best_D, mask
"""

if __name__=="__main__":
    # mask = Get_Mask([3, 4], [1, 2])
    # print(mask)
    
    Points = np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
                       [[-1, 2, 3], [-2, 3, 4], [-3, 4, 5], [-4, 5, 6]],
                       [[1, 2, -3], [2, 3, -4], [3, 4, -5], [4, 5, -6]]])
    Points = Points.reshape(-1, 3)
    print(Points)