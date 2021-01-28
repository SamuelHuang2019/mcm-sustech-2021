import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])
print(A)
print()
print(B)
print()
C1 = A.dot(B)
C2 = np.multiply(A, B)
C3 = A * B
C4 = A ** B  # exponential
C5 = A @ B

print("C1 =", C1)
print("C2 =", C2)
print("C3 =", C3)
print("C4 =", C4)
print("C5 =", C5)
