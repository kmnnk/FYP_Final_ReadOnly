import numpy as np
from scipy.stats import kruskal

########################################
#2D 58 vs 3D 58
auc_2d_58 = [0.5, 0.51, 0.57, 0.55, 0.51, 0.58, 0.61, 0.52, 0.59, 0.59, 0.52, 0.54, 0.62]
auc_3d_58 = [0.5, 0.58, 0.53, 0.56, 0.57, 0.5, 0.5, 0.5, 0.5, 0.56]

# Perform the Kruskal-Wallis test
statistic, p_value = kruskal(auc_2d_58, auc_3d_58)
print(print(f"Kruskal-Wallis Test: stat = {statistic}, p-value = {p_value:.4f}"))
########################################

#2D 348 vs 3D 348
auc_2d_348 = [0.65, 0.5, 0.61, 0.5, 0.5, 0.57, 0.67, 0.64, 0.61, 0.62, 0.64, 0.64, 0.63]
auc_3d_348 = [0.5, 0.5, 0.64, 0.68, 0.64, 0.7, 0.5]

statistic, p_value = kruskal(auc_2d_348, auc_3d_348)
print(print(f"Kruskal-Wallis Test: stat = {statistic}, p-value = {p_value:.4f}"))
########################################

# 58 vs 348
auc_2d_58 = [0.5, 0.51, 0.57, 0.55, 0.51, 0.58, 0.61, 0.52, 0.59, 0.59, 0.52, 0.54, 0.62]
auc_2d_348 = [0.65, 0.5, 0.61, 0.5, 0.5, 0.57, 0.67, 0.64, 0.61, 0.62, 0.64, 0.64, 0.63]

auc_3d_58 = [0.5, 0.58, 0.53, 0.56, 0.57, 0.5, 0.5, 0.5, 0.5, 0.56]
auc_3d_348 = [0.5, 0.5, 0.64, 0.68, 0.64, 0.7, 0.5]

statistic, p_value = kruskal(auc_2d_58, auc_2d_348)
print(print(f"Kruskal-Wallis Test: stat = {statistic}, p-value = {p_value:.4f}"))

statistic1, p_value1 = kruskal(auc_3d_58, auc_3d_348)
print(print(f"Kruskal-Wallis Test: stat = {statistic1}, p-value = {p_value1:.4f}"))
########################################

#AUC 2D across each dimension
twoD_1 = [0.5, 0.65]
twoD_2 = [0.51, 0.5]
twoD_3 = [0.57, 0.61]
twoD_4 = [0.55, 0.5]
twoD_5 = [0.51, 0.5]
twoD_6 = [0.58, 0.57]
twoD_7 = [0.61, 0.67]
twoD_8 = [0.52, 0.64]
twoD_9 = [0.59, 0.61]
twoD_10 = [0.59, 0.62]
twoD_11 = [0.52, 0.64]
twoD_12 = [0.54, 0.64]
twoD_13 = [0.62, 0.63]

#AUC 3D across each dimension that produced esults equally
threeD_1 = [0.5, 0.5]   
threeD_2 = [0.58, 0.5]
threeD_3 = [0.53, 0.64]
threeD_4 = [0.56, 0.68]
threeD_5 = [0.57, 0.64]
threeD_6 = [0.5, 0.7]
threeD_10 = [0.56, 0.5]

#dimensions 
two = twoD_2 + threeD_1
three = twoD_3 + threeD_2
four = twoD_4 + threeD_3
five = twoD_5 + threeD_4
six = twoD_6 + threeD_5
seven = twoD_12 + threeD_10


statistic, p_value = kruskal(one, two, three, four, five, six, seven)
print(print(f"Kruskal-Wallis Test: stat = {statistic}, p-value = {p_value:.4f}"))
########################################