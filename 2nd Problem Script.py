

import gauleg as gl 
import sympy as sp 
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import math

a = 0 
b = 1 



def a0(x):
    return 2 + math.sin(x)
# def a0(x):
#     if 0 <= x < 1/3: 
#         return 1.0 
#     elif 1/3 <= x <= 1:
#         return 2.0

def GL(xi, ci, y, x, n2, func):
	sum = 0 
	for i in range(n2):
		x_val= 0.5 * ((y - x) * xi[i] + (y + x))
		prod = ci[i] * func(x_val)
		sum += prod 
	final = (y - x) / 2 * sum
	return final

def phiij(numofnodes, i, j, l, xlist):
	finalsum = 0
	for k in range(numofnodes-1):

		x_left = xlist[k]
		x_right = xlist[k+1]
	 
		if k == i :
			d_phi_i = 2**l 
		elif k == i + 1 :
			d_phi_i = -2**l 
		else: 
			d_phi_i = 0 
 
		if k == j : 
			d_phi_j = 2**l 
		elif k == j + 1 :
			d_phi_j = -2**l 
		else:
			d_phi_j = 0 

		n2 = 5 
		xi = gl.gauleg(n2)[0]
		ci = gl.gauleg(n2)[1]
		onegl = GL(xi = xi, ci = ci, y = x_right, x = x_left, n2 = 5, func = a0) * d_phi_i *d_phi_j
		finalsum = finalsum + onegl

	return finalsum



def matrix(ijlist, matrix, n1, l, xlist):
	for i in ijlist:
		for j in ijlist:
			aij = phiij(numofnodes=n1, i=i, j =j, l = l, xlist=xlist)
			matrix[i,j] = aij 

	return matrix 



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
        x_left = xlist[k]
        x_right = xlist[k+1]
        sum = 0
        for j in range(n2):
            s = 0.5 * ((x_right - x_left) * xi[j] + (x_right + x_left))
            h = f(s) * phi_j_func(s)
            prod = ci[j] * h
            sum += prod
        onegl = (x_right - x_left) / 2 * sum
        finalsum += onegl

    return finalsum

def f(s): 
    return -4 + math.cos(s)  - 2*s*math.cos(s) - 2* math.sin(s)
# def f(s):
#     if 0 <= s < 1/3:
#         return 2.0
#     elif 1/3 <= s <= 1:
#         return 4.0

def finding_F(F_empty, ijlist, xlist, l, n2, n1):
    xi = gl.gauleg(n2)[0]
    ci = gl.gauleg(n2)[1]
    for i in ijlist:
        Fj = intergrand(numofnodes=n1, i = i, l = l, xlist=xlist, f= f, ci=ci, xi = xi, n2 = n2)
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

    A = matrix(ijlist = ijlist, matrix= matrixA, n1 = n1, l = l,xlist= x_list)
    print("A:",A)
    F = finding_F(F_empty=matrixF, ijlist=ijlist, xlist=x_list, l = l, n2 = n2, n1 = n1)
    print("F:", F)
    c = inv(matrix =A, F = F )
    print("c:",  c)

    approximation = approx_new(ijlist=ijlist, c=c, F =F)
    print(approximation)
    error = actual - approximation
    abserr = abs(error)
    logerror = math.log(abserr)

    return error

# print(oneerror(5))
# print(oneerror(4))
# print(oneerror(3))


#print(oneerror(6))

# l_list = list(range(6))
# def errordf(llist):
#     error = []
#     for l in llist:
#         logerror = oneerror(l)
#         error.append({'l': l, 'log error': logerror})
    
#     df_errors = pd.DataFrame(error)
#     return df_errors

# dataframe = errordf(l_list)
# plt.figure(figsize=(16, 12))
# plt.plot(dataframe['l'], dataframe['log error'], marker='o', linestyle='-')
# plt.xlabel('l')
# plt.ylabel('log error')
# plt.title('Plot of log error against l')
# plt.grid(True)
# plt.show()

def exact_solution(x):
    return x*(1.0 - x)
x_fine = np.linspace(0.0, 1.0, 200)
u_exact = exact_solution(x_fine)

plt.figure(figsize=(6,4))

def sol(l):
    h = 2 **(-l)
    n1 = 2**l + 1
    ijlist = list(range(2**l - 1))
    x_list = gl.listi(a, b, h, n1)
    dim = 2**l -1
    matrixF = sp.zeros(dim, 1)
    matrixA= sp.zeros(dim, dim)
    n2 = 5

    A = matrix(ijlist = ijlist, matrix= matrixA, n1 = n1, l = l,xlist= x_list)
    print("A:",A)
    F = finding_F(F_empty=matrixF, ijlist=ijlist, xlist=x_list, l = l, n2 = n2, n1 = n1)
    print("F:", F)
    c = inv(matrix =A, F = F )
    print("c:",  c)
    return -c, x_list

def assemble_nodal_values(C):
    """
    Given:
      C = [U_1, U_2, ..., U_{N-1}]  from the system solve
    Returns:
      U_full = [U_0, U_1, U_2, ..., U_{N-1}, U_N]  (with BCs)
    """
    N_minus_1 = len(C)          # number of interior unknowns
    U_full = [0]*(N_minus_1+2)  # +2 for endpoints
    U_full[0] = 0.0            # Dirichlet BC at x_0
    U_full[-1] = 0.0           # Dirichlet BC at x_N
    
    # Fill in interior:
    for i in range(N_minus_1):
        U_full[i+1] = C[i]
    
    return U_full

c_sol, mesh = sol(6)
nodal = assemble_nodal_values(c_sol)
x_nodal = np.array(mesh, dtype=float)
u_nodal = np.array(nodal, dtype=float)

# 1) Plot the piecewise-linear solution
plt.plot(
    x_nodal,
    u_nodal,
    marker='o',
    linestyle='-',
    color='blue',
    label='Numerical (Refined Mesh)'
)

# 2) Plot the exact solution as a smooth curve
plt.plot(
    x_fine,
    u_exact,
    color='red',
    linewidth=2,
    label='Exact: x(1-x)'
)

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Refined Numerical Solution vs. Exact')
plt.grid(True)
plt.legend()
plt.show()
