##################################################################################
#
# PROJECT: Optimization Research of Epidemic Control Resource Distribution Network
# COURSE : General Optimization of Chemical Engineering Systems
#
# MODULE : Model Build & Test Run
#
# AUTHORS: Shuaiyu Xiang, Yiming Bai.
#          Department of Chemical Engineering, Tsinghua University, P. R. China
#
##################################################################################

# --------------------------------------------------------------------------------
# Update Information
# --------------------------------------------------------------------------------
_module_last_updated = "2020-11-03"

import gwo
import math
import pandas as pd
import quartzclock as qc
from numpy import argmin
from copy import deepcopy
from scipy.optimize import minimize

"""
如何使用灰狼优化算法:

先定义目标函数(min)，def xxx(args):, 自变量以list形式输入，返回函数值

gwo.optimize(xxx, lower_bounds, upper_bounds, size=100, max_iter=100, show=True)
             xxx是目标函数名;
             lower_bounds和upper_bounds分别是自变量的下限和上限, 以list形式输入;
             size是种群中狼的数目, 默认100;
             max_iter是最大迭代次数, 默认100;
             show为True时输出每步迭代的最小函数值。                
"""
df = pd.read_csv('paths.csv', index_col=0, header=0)
n_vector = {}
for index, row in df.iterrows():
    n_vector[int(index)] = (row[0], row[1], row[2])

# n_vector = {1: (4.48, 1.43, 0),
#             2: (7.51, 0.89, 0),
#             3: (10.36, 0.89, 0),
#             4: (14.68, 0.24, 0),
#             5: (16.63, 0.24, 0),
#             6: (11.34, 4.89, 0),
#             7: (13.17, 2.89, 0),
#             8: (15.9, 2.48, 0),
#             9: (17.31, 2.48, 0),
#             10: (19.82, 2.75, 0),
#             11: (21.68, 5.36, 0),
#             12: (7.95, 6.33, 0),
#             13: (9.31, 7.74, 0),
#             14: (11.58, 7.55, 0),
#             15: (14.68, 8.01, 0),
#             16: (22.82, 8.7, 0),
#             17: (3.4, 10.38, 0),
#             18: (6.71, 9.36, 0),
#             19: (9.97, 10.43, 0),
#             20: (12.22, 10.32, 0),
#             21: (14.62, 10.43, 0),
#             22: (0.47, 14.28, 0),
#             23: (3.4, 13.66, 0),
#             24: (15.21, 13.37, 0),
#             25: (9.52, 17.35, 0),
#             26: (9.79, 20.07, 0),
#             27: (12.4, 17.83, 0),
#             28: (14.94, 18.2, 0),
#             29: (18.84, 20.56, 0),
#             30: (21.67, 17.91, 4),
#             31: (16.84, 26.19, 4),
#             32: (21.88, 25.26, 4),
#             33: (16.01, 29.16, 4)}

m_h = (1.5, 17.02)

l_max = 5.8
l_min = 1

h_min = 1/1000
q_min = 100

C_s = 4
C_d = 0.9
k_s = 0.1
k_d = 0.04

u, v, w = 0.7, 0.2, 0.1

def get_distance(i, j):
    return math.sqrt((i[0] - j[0])** 2 + (i[1] - j[1])** 2)


def hardness(points):
    m_vector = {1: (points[0], points[1]),
                2: (points[2], points[3]),
                3: (points[4], points[5]),
                4: (points[6], points[7]),
                5: (points[8], points[9])}
    
    n_index = deepcopy(n_vector)

    for i, n in n_vector.items():
        distances = []

        for m in m_vector.values():
            distances.append(get_distance(m, n[0:2]))
        min_dist = min(distances)
        index = argmin(distances) + 1

        n_index[i] = (index, min_dist)
        
    def H():
        H_sum = 0
        
        for i, n in n_vector.items():
            H_sum += n[2] * h_min * math.exp(n_index[i][1] / l_max) if n_index[i][1] > l_max else n[2] * h_min
        
        return H_sum

    def S():
        storages = []
        
        for i in m_vector:
            storage = 0
            for ix, n in n_vector.items():
                if n_index[ix][0] == i:
                    storage += 1.2 * n[2]
            storages.append(storage)
        
        S_sum = sum(map(lambda x: C_s + k_s * (x / q_min)**(2 / 3), storages))
        
        return (S_sum, storages)

    def D():
        D_sum = 0

        for m in m_vector.values():
            D_sum += C_d + k_d * (get_distance(m, m_h) / l_min)
        
        return D_sum
    
    return u * H() + v * S()[0] + w * D()


condition = [{'type':'ineq', 'fun':lambda x:x[0] - 4},
             {'type': 'ineq', 'fun': lambda x: x[2] - 4},
             {'type': 'ineq', 'fun': lambda x: x[4] - 4},
             {'type': 'ineq', 'fun': lambda x: x[6] - 4},
             {'type': 'ineq', 'fun': lambda x: x[8] - 4},
             {'type': 'ineq', 'fun': lambda x: 30 - x[0]},
             {'type': 'ineq', 'fun': lambda x: 30 - x[2]},
             {'type': 'ineq', 'fun': lambda x: 30 - x[4]},
             {'type': 'ineq', 'fun': lambda x: 30 - x[6]},
             {'type': 'ineq', 'fun': lambda x: 30 - x[8]},
             {'type': 'ineq', 'fun': lambda x: x[1]},
             {'type': 'ineq', 'fun': lambda x: x[3]},
             {'type': 'ineq', 'fun': lambda x: x[5]},
             {'type': 'ineq', 'fun': lambda x: x[7]},
             {'type': 'ineq', 'fun': lambda x: x[9]},
             {'type': 'ineq', 'fun': lambda x: 21 - x[1]},
             {'type': 'ineq', 'fun': lambda x: 21 - x[3]},
             {'type': 'ineq', 'fun': lambda x: 21 - x[5]},
             {'type': 'ineq', 'fun': lambda x: 21 - x[7]},
             {'type': 'ineq', 'fun': lambda x: 21 - x[9]}]
# gwo.optimize(hardness, [0] * 10, [30] * 10, size=100, max_iter=100)

# def sq(x):
#     return x[0]** 2
    
# gwo.optimize(sq, [-2], [3], size=100, max_iter=50)


x0 = [14, 10] * 5

qc.start_timing()
result = minimize(hardness, x0, method='COBYLA', constraints=condition)
qc.end_timing()

print(result.fun, result.success, result.x, sep='\n')
