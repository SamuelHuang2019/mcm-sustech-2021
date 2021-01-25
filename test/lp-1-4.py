import numpy as np
from scipy.optimize import linprog

# values
c = -np.array([3100] * 3 + [3800] * 3 + [3500] * 3 + [2850] * 3)
print(c)
print()

# restrictions
A_ub = np.zeros((4 + 3 + 3, 12))
b_ub = np.zeros(4 + 3 + 3)

# total weight of loads
for i in range(4):
    A_ub[i, i * 3:(i + 1) * 3] = 1
b_ub[:4] = 18, 15, 23, 12

# weight of loads in each sector
wt = 18, 15, 23, 12
for i in range(3):
    A_ub[4 + i, range(i, 12, 3)] = 1
b_ub[4:4 + 3] = 10, 16, 8

# volume
# v = 480 / 18, 650 / 15, 580 / 23, 390 / 12
v = 480, 650, 580, 390
for i in range(3):
    A_ub[4 + 3 + i, range(i, 12, 3)] = v
b_ub[4 + 3:4 + 3 + 3] = 6800, 8700, 5300
print(A_ub)
print()
print(b_ub)
print()

# equalities
A_eq = np.zeros((2, 12))
A_eq[:, range(0, 12, 3)] = 1 / 10
A_eq[0, range(1, 12, 3)] = -1 / 16
A_eq[1, range(2, 12, 3)] = -1 / 8
print(A_eq)
print()
b_eq = np.zeros(2)
print(b_eq)
print()

# linear programming
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
print(result)
print()

x = np.array(result.x)
loads = []
for i in range(4):
    loads += [sum(x[i:12:3])]
print(loads)
