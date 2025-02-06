import gauleg as gl 
import sympy as sp 
import numpy as np 
import pandas as pd 
import math
import matplotlib 
import matplotlib.pyplot as plt

def legendre_polynomial(n):
    y = sp.symbols('y')
    if n == 0:
        return sp.Lambda(y, 1)
    elif n == 1:
        return sp.Lambda(y, y)
    else:
        P_n_minus_1 = legendre_polynomial(n-1)
        P_n_minus_2 = legendre_polynomial(n-2)
        return sp.Lambda(y, ((2*n - 1) * y * P_n_minus_1(y) - (n - 1) * P_n_minus_2(y)) / n)
    

def Anm(n, m):
    y = sp.symbols('y')
    Ln = legendre_polynomial(n)
    Lm = legendre_polynomial(m)
    return 0 * Ln(y) * Lm(y)
    #    return y * Ln(y) * Lm(y) for a non deterministic case

def Anm_gauleg(xi, ci, b , a, n2, n, m):
    sum = 0 
    Anm_expr = Anm(n,m)
    for i in range(n2):
        y_value = (0.5*(((b-a)*xi[i])+(b+a)))
        prod = ci[i]* Anm_expr.subs(sp.symbols('y'), y_value)
        sum = sum + prod 
    return sum 

def A_matrix(n,m):
    A_matrix = sp.zeros(len(n),len(m))
    n2 = math.floor((len(n)+len(m))/2) + 1
    xi = gl.gauleg(n2)[0] # satisfying dop of GL for p+1 is 2p + 1
    ci = gl.gauleg(n2)[1]
    for ni in n:
        for mi in m:
            A_matrix[ni ,mi] = Anm_gauleg(xi= xi, ci = ci, b = 1, a = -1, n2 = n2, n = ni, m = mi)

    return A_matrix


def a0(x):
    if 0 <= x <= 1/3:
        return 1
    elif 1/3 < x <= 1:
        return 2

def a1(x):
    return 0 *x

def GL(xi, ci, x_right, x_left, n2, func):
	sum = 0 
	for i in range(n2):
		prod = ci[i]*func(0.5*(((x_right-x_left)*xi[i])+(x_right+x_left)))
		sum = sum + prod 
	return (x_right-x_left)/2 * (sum)

def piecewise_GL(integrand, x_left, x_right, n2):
    """
    Integrate 'integrand(x)' from x_left to x_right using Gauss-Legendre,
    but if [x_left, x_right] crosses x=1/3, split it into two parts:
      [x_left, 1/3]  and  [1/3, x_right]
    and sum the results.
    """
    discont = 1/3
    xi, ci = gl.gauleg(n2) 
    # If the entire interval is on one side of 1/3, do one normal G-L integral
    if x_right <= discont or x_left >= discont:
        return GL(xi =  xi, ci = ci, x_left=x_left, x_right=x_right, func=integrand, n2=n2)

    # Otherwise, we split at x=1/3
    # part1 = integral from x_left..1/3
    part1 = GL(xi =  xi, ci = ci, x_left=x_left, x_right=discont, func=integrand, n2=n2)
    # part2 = integral from 1/3..x_right
    part2 = GL(xi =  xi, ci = ci, x_left=discont, x_right=x_right, func=integrand, n2=n2)

    return part1 + part2

def dphi_i_on_element(i, k, xlist):
    """
    Return the (constant) slope of the i-th shape function on the k-th subinterval
    [ x_k, x_{k+1} ] in a 1D mesh with nodes x_0 < ... < x_N.
    """
    if i == k:
        # node i is the left endpoint of the subinterval => slope from 1 at x_k down to 0 at x_{k+1}
        dx = xlist[k+1] - xlist[k]
        return -1.0 / dx
    elif i == k+1:
        # node i is the right endpoint => slope from 0 at x_k up to 1 at x_{k+1}
        dx = xlist[k+1] - xlist[k]
        return +1.0 / dx
    else:
        return 0.0


def S1_ji(func, mesh):
    N = len(mesh)  # number of nodes
    S1_mat = sp.zeros(N, N)
    n2 = 5
    # Assemble diagonal entries S1[j,j]
    for j in range(N):
        diag_val = 0.0
        # Contribution from the left subinterval [mesh[j-1], mesh[j]]
        if j > 0:
            x_left = mesh[j-1]
            x_right = mesh[j]
            # On [mesh[j-1], mesh[j]], phi_j is the right (increasing) part.
            def integrand_left(x):
                # dphi_j/dx on [mesh[j-1], mesh[j]]:
                return func(x) * (dphi_i_on_element(j, j-1, mesh))**2
            diag_val += piecewise_GL(integrand_left, x_left, x_right, n2=n2)
            
        # Contribution from the right subinterval [mesh[j], mesh[j+1]]
        if j < N-1:
            x_left = mesh[j]
            x_right = mesh[j+1]
            # On [mesh[j], mesh[j+1]], phi_j is the left (decreasing) part.
            def integrand_right(x):
                # dphi_j/dx on [mesh[j], mesh[j+1]]:
                return func(x) * (dphi_i_on_element(j, j, mesh))**2
            diag_val += piecewise_GL(integrand_right, x_left, x_right, n2=n2)
            
        S1_mat[j, j] = diag_val

    for j in range(1, N):
        # Contribution for S1[j, j-1] (and by symmetry S1[j-1, j]) over [mesh[j-1], mesh[j]]
        x_left = mesh[j-1]
        x_right = mesh[j]
        def integrand_off(x):
            # For node j-1 on the element, phi_{j-1} is the left (decreasing) part,
            # and for node j, phi_j is the right (increasing) part.
            return func(x) * dphi_i_on_element(j-1, j-1, mesh) * dphi_i_on_element(j, j-1, mesh)
        val = piecewise_GL(integrand_off, x_left, x_right, n2=n2)
        S1_mat[j, j-1] = val
        S1_mat[j-1, j] = val  # Exploiting symmetry
        
    return S1_mat

def S0_ji(func, mesh):
    N = len(mesh)
    S0_mat = sp.zeros(N, N)
    n2 = 10
    # assembly of diagonal entries of S0 
    for j in range(N):
            diag_val = 0.0
            # Contribution from the left subinterval [mesh[j-1], mesh[j]]
            if j > 0:
                x_left = mesh[j-1]
                x_right = mesh[j]
                # On [mesh[j-1], mesh[j]], phi_j is the right (increasing) part.
                def integrand_left(x):
                    # dphi_j/dx on [mesh[j-1], mesh[j]]:
                    return func(x) * (dphi_i_on_element(j, j-1, mesh))**2
                diag_val += piecewise_GL(integrand_left, x_left, x_right, n2=n2)
                
            # Contribution from the right subinterval [mesh[j], mesh[j+1]]
            if j < N-1:
                x_left = mesh[j]
                x_right = mesh[j+1]
                # On [mesh[j], mesh[j+1]], phi_j is the left (decreasing) part.
                def integrand_right(x):
                    # dphi_j/dx on [mesh[j], mesh[j+1]]:
                    return func(x) * (dphi_i_on_element(j, j, mesh))**2
                diag_val += piecewise_GL(integrand_right, x_left, x_right, n2=n2)
                
            S0_mat[j, j] = diag_val

    for j in range(1, N):
        # Contribution for S1[j, j-1] (and by symmetry S1[j-1, j]) over [mesh[j-1], mesh[j]]
        x_left = mesh[j-1]
        x_right = mesh[j]
        def integrand_off(x):
            # For node j-1 on the element, phi_{j-1} is the left (decreasing) part,
            # and for node j, phi_j is the right (increasing) part.
            return func(x) * dphi_i_on_element(j-1, j-1, mesh) * dphi_i_on_element(j, j-1, mesh)
        val = piecewise_GL(integrand_off, x_left, x_right, n2=n2)
        S0_mat[j, j-1] = val
        S0_mat[j-1, j] = val  # Exploiting symmetry
        
    return S0_mat


def A_S1(matrixA, matrixS1):
    A_S1 = sp.kronecker_product(matrixA, matrixS1)
    return A_S1

def delta_mn(m,n):
    if n == m:
        return 1 
    else:
        return 0 

def delta(n,m):
    delta = sp.zeros(len(n), len(m))
    for ni in n:
        for mi in m: 
            delta[ni,mi] = delta_mn(ni,mi)

    return delta 

def S0_delta(matrixS0, matrixdelta):
    return sp.kronecker_product(matrixS0 , matrixdelta) 

def S_in_jm(matrixS1_Amn, matrixS0_delta):
    return matrixS0_delta + matrixS1_Amn


def GL(xi, ci, x_left, x_right, n2, func):
    """
    Performs a Gauss–Legendre integration on [x_left, x_right] of func(x).
    """
    ssum = 0.0
    for i in range(n2):
        # Map the Gauss–Legendre node from [-1,1] to [x_left,x_right]
        x_mapped = 0.5 * ((x_right - x_left)*xi[i] + (x_right + x_left))
        ssum += ci[i] * func(x_mapped)
    return 0.5 * (x_right - x_left) * ssum

def piecewise_GL(integrand, x_left, x_right, n2):
    """
    Integrate integrand(x) from x_left to x_right using Gauss-Legendre.
    If the interval crosses x=1/3, split it into [x_left, 1/3] and [1/3, x_right].
    """
    discont = 1/3
    xi, ci = gl.gauleg(n2)
    
    if x_right <= discont or x_left >= discont:
        return GL(xi=xi, ci=ci, x_left=x_left, x_right=x_right, func=integrand, n2=n2)
    else:
        part1 = GL(xi=xi, ci=ci, x_left=x_left, x_right=discont, func=integrand, n2=n2)
        part2 = GL(xi=xi, ci=ci, x_left=discont, x_right=x_right, func=integrand, n2=n2)
        return part1 + part2

def phi_i(i, x, mesh):
    """
    Standard 1D hat (finite element) function.
    This returns the value of the i-th hat function at x, given a mesh.
    """
    N = len(mesh) - 1  # number of elements
    # For node 0 (left boundary)
    if i == 0:
        if x < mesh[0] or x > mesh[1]:
            return 0.0
        else:
            return (mesh[1] - x) / (mesh[1] - mesh[0])
    # For node N (right boundary)
    elif i == N:
        if x < mesh[N-1] or x > mesh[N]:
            return 0.0
        else:
            return (x - mesh[N-1]) / (mesh[N] - mesh[N-1])
    # For interior nodes
    else:
        if x < mesh[i-1] or x > mesh[i+1]:
            return 0.0
        elif x <= mesh[i]:
            return (x - mesh[i-1]) / (mesh[i] - mesh[i-1])
        else:
            return (mesh[i+1] - x) / (mesh[i+1] - mesh[i])

def build_force_vector(f, mesh, n2=5):
    """
    Assemble the force (load) vector F where
         F[i] = ∫ f(x) * phi_i(x) dx,
    computed by integrating over the support of the i-th finite element function.
    
    Parameters:
      f     : The source function, f(x).
      mesh  : A list of node coordinates, e.g. [x0, x1, ..., xN].
      n2    : Number of Gauss-Legendre points.
      
    Returns:
      F     : A column vector (Sympy Matrix) of size (N+1) x 1.
    """
    num_nodes = len(mesh)  # = N+1
    F = sp.zeros(num_nodes, 1)
    
    # Loop over each finite element function phi_i.
    for i in range(num_nodes):
        total = 0.0
        
        # The support of phi_i:
        # For interior nodes, the support is [mesh[i-1], mesh[i+1]] split into two intervals.
        # For the left boundary, support is [mesh[0], mesh[1]].
        # For the right boundary, support is [mesh[-2], mesh[-1]].
        
        # Left subinterval (if it exists)
        if i > 0:
            x_left  = mesh[i-1]
            x_right = mesh[i]
            def integrand_left(x):
                return f(x) * phi_i(i, x, mesh)
            total += piecewise_GL(integrand_left, x_left, x_right, n2=n2)
        
        # Right subinterval (if it exists)
        if i < num_nodes - 1:
            x_left  = mesh[i]
            x_right = mesh[i+1]
            def integrand_right(x):
                return f(x) * phi_i(i, x, mesh)
            total += piecewise_GL(integrand_right, x_left, x_right, n2=n2)
        
        F[i, 0] = total
    return F

def f(s):
    if 0 <= s <= 1/3:
        return 2.0
    elif 1/3 < s <= 1:
        return 4.0
    

def extract_interior(S, F):
    S_int = S[1:-1, 1:-1]
    F_int = F[1:-1, :]

    return S_int, F_int

def c(S_int, F_int):
        c = S_int.LUsolve(F_int)
        return c

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
mesh = gl.listi(0,1, 2**(-1), 2**1 + 1)
S0 = S0_ji(func = a0, mesh = mesh)
fvect = build_force_vector(f, mesh, 5)
S0_int, f_int = extract_interior(S0, fvect)
c_sol = c(S0_int, f_int)
nodal = assemble_nodal_values(c_sol)
x_nodal = np.array(mesh, dtype=float)
u_nodal = np.array(nodal, dtype=float)


print("S0:", S0)
print("S0_int:", S0_int)
print("Full F:", fvect)
print("F_int:", f_int)
def exact_solution(x):
    return x*(1.0 - x)

print(u_nodal)
# 1) Plot the piecewise-linear solution
plt.plot(
    x_nodal,
    u_nodal,
    marker='o',
    linestyle='-',
    color='blue',
    label='Numerical (Refined Mesh)'
)
# We'll plot the exact solution on a fine grid from 0..1
x_fine = np.linspace(0.0, 1.0, 200)
u_exact = exact_solution(x_fine)

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