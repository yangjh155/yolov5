import numpy as np
import matplotlib.pyplot as plt
from Plane_ransac import Calculate_Theta

data = np.loadtxt("./Plane0.txt")

alpha = 0.07

method = "ransac"

A_idx, B_idx, D_idx, theta_idx = 0, 1, 2, 3
if method == "lr":
    A_idx += 4
    B_idx += 4
    D_idx += 4
    theta_idx += 4

for i in range(data.shape[0]):
    if i == 0:
        smooth_A = [data[i,A_idx]]
        smooth_B = [data[i,B_idx]]
        smooth_D = [data[i,D_idx]]
        smooth_theta = [data[i,theta_idx]]
        theta = [Calculate_Theta(smooth_A[-1], smooth_B[-1], 1)]
    else:
        smooth_A += [alpha * data[i][A_idx] + (1 - alpha) * smooth_A[-1]]
        smooth_B += [alpha * data[i][B_idx] + (1 - alpha) * smooth_B[-1]]
        smooth_D += [alpha * data[i][D_idx] + (1 - alpha) * smooth_D[-1]]
        smooth_theta += [alpha * data[i][theta_idx] + (1 - alpha) * smooth_theta[-1]]
        theta += [Calculate_Theta(smooth_A[-1], smooth_B[-1], 1)]   
        
        
plt.subplot(2,2,1)    
plt.scatter(range(data.shape[0]), data[:,0], c='r', s=3)
plt.plot(smooth_A)
plt.title("A")

plt.subplot(2,2,2)
plt.scatter(range(data.shape[0]), data[:,1], c='r', s=3)
plt.plot(smooth_B)
plt.title("B")

plt.subplot(2,2,3)
plt.scatter(range(data.shape[0]), data[:,2], c='r', s=3)
plt.plot(smooth_D)
plt.title("D")

plt.subplot(2,2,4)
plt.scatter(range(data.shape[0]), data[:,3], c='r', s=3)
plt.plot(smooth_theta)
plt.title("theta")
plt.savefig(f"{method}.png")
plt.show()