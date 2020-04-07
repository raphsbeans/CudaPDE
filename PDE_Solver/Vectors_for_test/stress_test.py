# -*- coding: utf-8 -*-
"""
This script will create a series of matrix to check if the algorithm is right
To call this script from terminal you need to do
python .\Vectors_for_test\stress_test.py Dim Size
"""

import numpy as np
import sys

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def save(array, path):
    main_path = 'Vectors_for_test/'
    np.savetxt(main_path + path + '.txt', array)

def save_all():
    save(a, 'a')
    save(b, 'b')
    save(c, 'c')
    save(x, 'x')
    save(y, 'y')

Dim = int(sys.argv[1])
Size = int(sys.argv[2])

print('Initializing script for ' + str(Size) + ' different systems of dimention = ' + str(Dim))

x = np.random.rand(Size, Dim)
a,b,c,y = [],[],[],[]
i = 0
while i < Size:
    aux_a = np.random.rand(Dim)
    aux_b = np.random.rand(Dim)
    aux_c = np.random.rand(Dim)
    if i%100 == 0:
        print(i)
    A = tridiag(aux_a[1:Dim], aux_b, aux_c[0:Dim-1])
    if (abs(np.linalg.det(A)) < 1e-3):
        print(i)
        print("NON SINGULAR!! Trying again!")
        continue

    aux_a[0] = 0
    aux_c[-1] = 0
    y.append(np.matmul(A, x[i]))
    a.append(aux_a)
    b.append(aux_b)
    c.append(aux_c)
    i += 1

y = np.array(y)
a = np.array(a)
b = np.array(b)
c = np.array(c)

save_all()
