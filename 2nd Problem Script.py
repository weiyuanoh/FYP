

import gauleg as gl 
import sympy as sp 
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import math

a = 0 
b = 1 



def intergrand(numofnodes, i, l, xlist, f, ci, xi, n2):
    def phi_j(numofnodes, i, l, k):
        def phi_j_function(s):
            if k == i:
                return 2**l * (s - (i / 2**l))
            elif k == i + 1:
                return -(2**l * (s - ((i + 2) / 2**l)))
            else:
                return 0
        return phi_j_function

    finalsum = 0
    for k in range(numofnodes-1):
        phi_j_func = phi_j(numofnodes, i, l, k)
        x = xlist[k]
        y = xlist[k+1]
        sum = 0
        for j in range(n2):
            s = 0.5 * ((y - x) * xi[j] + (y + x))
            h = f(s) * phi_j_func(s)
            prod = ci[j] * h
            sum += prod
        onegl = (y - x) / 2 * sum
        finalsum += onegl

    return finalsum

def finding_F(F_empty, ijlist, xlist, l, n2, n1):
    xi = gl.gauleg(n2)[0]
    ci = gl.gauleg(n2)[1]
    for i in ijlist:
        Fj = intergrand(numofnodes=n1, i = i, l = l, xlist=xlist, f= gl.f, ci=ci, xi = xi, n2 = n2)
        F_empty[i,0] = Fj 
    return F_empty




def inv(matrix, F):  
    a_inv = matrix.inv()
    c = a_inv * F
    return c

def approx_new(ijlist, c, F):
    approx_new = 0 
    for i in ijlist : 
        prod = c[i] * F[i]
        approx_new = approx_new + prod
    return approx_new

def oneerror(l):
    actual = 0.8146667469752307

    h = 2 **(-l)
    n1 = 2**l + 1
    ijlist = list(range(2**l - 1))
    x_list = gl.listi(a, b, h, n1)
    dim = 2**l -1
    matrixF = sp.zeros(dim, 1)
    matrixA= sp.zeros(dim, dim)
    n2 = 5

    A = gl.matrix(ijlist = ijlist, matrix= matrixA, n1 = n1, l = l,xlist= x_list)
    F = finding_F(F_empty=matrixF, ijlist=ijlist, xlist=x_list, l = l, n2 = n2, n1 = n1)
    c = inv(matrix =A, F = F )
    approximation = approx_new(ijlist=ijlist, c=c, F =F)
    error = actual - approximation
    abserr = abs(error)
    logerror = math.log(abserr)

    return  F


print(oneerror(4))


'''
print(oneerror(5))
print(oneerror(4))
print(oneerror(3))
l_list = list(range(9))
def errordf(llist):
    error = []
    for l in llist:
        logerror = oneerror(l)
        error.append({'l': l, 'log error': logerror})
    
    df_errors = pd.DataFrame(error)
    return df_errors

dataframe = errordf(l_list)
plt.figure(figsize=(16, 12))
plt.plot(dataframe['l'], dataframe['log error'], marker='o', linestyle='-')
plt.xlabel('l')
plt.ylabel('log error')
plt.title('Plot of log error against l')
plt.grid(True)
plt.show()
'''



