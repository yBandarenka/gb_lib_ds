
#Задание 1

import numpy as np

a = np.array([[1, 2, 3, 3, 1], [6, 8, 11, 10, 7]]).transpose()

print("a:\n")
print(a)

mean_a = np.mean(a, 0)

print("\nmean_a:\n")
print(mean_a)

#Задание 2

a_centered = a - mean_a

print("\na_centered:\n")
print(a_centered)

#Задание 3

a_centered_sp = a_centered.T[0] @ a_centered.T[1]

a_centered_sp_2 = a_centered_sp / (a_centered.shape[0] - 1)

print("\na_centered_sp:\n")
print(a_centered_sp)
print("\na_centered_sp_2:\n")
print(a_centered_sp_2)

#Задание 4**

f_cov = np.cov(a.T)[0, 1]

print("\ncov (version 2):\n")
print(f_cov)