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



def a(x, y):
    return 2 + math.sin(x)


# def a0(x):
#     return 2 + math.sin(x)

# def a0(x):
#     if 0 <= x <= 0.2: 
#         return 1.0 
#     elif 0.2 <= x <= 1:
#         return 2.0

def a0(x):
    if 0 <= x <= 0.1:
        return 1
    elif 0.1 <= x <= 1:
        return 2

# def a0(x): 
#     if 0<= x < 0.75:
#         return 1 
#     elif 0.75<= x <= 1:
#         return 2

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
    discont = 0.1
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


def phiij(numofnodes, i, j, xlist, func):
    finalsum = 0
    for k in range(numofnodes-1):
        x_left = xlist[k]
        x_right = xlist[k+1]

        # Derivatives of shape functions phi_i, phi_j on element k
        d_phi_i = dphi_i_on_element(i, k, xlist)
        d_phi_j = dphi_i_on_element(j, k, xlist)
        def local_integrand(x_val):
            return func(x_val) * (d_phi_i * d_phi_j)

        # Now call piecewise_GL with local_integrand
        val = piecewise_GL(local_integrand, x_left, x_right, n2=5)
        finalsum += val
    return finalsum

def S1(func, mesh):
    I = list(range(1, len(mesh)-1))
    S1 = sp.zeros(len(I), len(I))
    target = func
    for local_i, g_i in enumerate(I):
            for local_j, g_j in enumerate(I):
                # 4) Evaluate your shape-function integral
                S1[local_i, local_j] = phiij(
                    numofnodes=len(mesh), 
                    i=g_i, 
                    j=g_j,
                    xlist=mesh,
                    func=func
                )

    return S1

def S0(func, mesh):
    I=list(range(1,len(mesh)-1))
    S0 = sp.zeros(len(I), len(I))
    target = func
    for local_i, g_i in enumerate(I):
            for local_j, g_j in enumerate(I):
                # 4) Evaluate your shape-function integral
                S0[local_i, local_j] = phiij(
                    numofnodes=len(mesh), 
                    i=g_i, 
                    j=g_j,
                    xlist=mesh,
                    func= target
                )

    return S0


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


def phi(i, s, xlist):
    """
    Evaluate the piecewise-linear hat function phi_i at point s,
    given the node array xlist = [x_0, x_1, ..., x_N].
    """
    # Find which interval [x_k,x_{k+1}] contains s
    # If s < x_0 or s > x_N, phi_i(s)=0 outside domain
    if s < xlist[0] or s > xlist[-1]:
        return 0.0

    # Find k: x_k <= s < x_{k+1}
    k_found = None
    for k in range(len(xlist)-1):
        if xlist[k] <= s < xlist[k+1]:
            k_found = k
            break
    # handle s == xlist[-1]
    if s == xlist[-1]:
        k_found = len(xlist)-2

    # Cases:
    # 1) If i not in {k,k+1}, phi_i(s)=0 in that subinterval
    if i != k and i != k+1:
        return 0.0

    # 2) If i == k, then phi_i goes from 1 at x_k to 0 at x_{k+1}
    if i == k:
        dx = xlist[k+1] - xlist[k]
        # linear from phi_i(x_k)=1 to phi_i(x_{k+1})=0
        # slope = -1/dx
        return 1.0 - (s - xlist[k]) / dx

    # 3) If i == k+1, then phi_i goes from 0 at x_k to 1 at x_{k+1}
    elif i == k+1:
        dx = xlist[k+1] - xlist[k]
        return (s - xlist[k]) / dx
    
def integrand_for_force(i, xlist, f, xi, ci, n2):
    """
    Compute \int f(s)*phi_i(s) ds over the entire domain,
    by summing subinterval integrals or a single Gauss-Legendre over [0,1].
    But typically we do subinterval-based integration for a non-uniform mesh.
    """
    # We'll do an element-by-element integration approach:
    total = 0.0
    for k in range(len(xlist)-1):
        x_left = xlist[k]
        x_right = xlist[k+1]
        # do Gauss-Legendre on each subinterval
        sum_local = 0.0
        for j in range(n2):
            s = 0.5*((x_right - x_left)*xi[j] + (x_left + x_right))
            w = ci[j]
            val = f(s)*phi(i, s, xlist)
            sum_local += w*val
        total += 0.5*(x_right - x_left)*sum_local
    return total

# def f(s): 
#     return 2* (2 + math.sin(s)) - math.cos(s) *(1 -2*s)

# def f(s):
#     if 0 <= s < 0.2:
#         return 2.0
#     elif 0.2 <= s <= 1:
#         return 4.0

def f(s):
    if 0 <= s < 0.1:
        return 2
    elif 0.1 <= s <= 1:
        return 4
    
# def f(s):
#     if 0 <= s < 0.75:
#         return 2
#     elif 0.75 <= s <= 1:
#         return 4


def build_force_vector2(num_nodes, xlist, f):
    """
    Build the interior force vector F where
      F[i] = ∫ f(x)*phi_i(x) dx
    skipping boundary shape functions (i=0, i=N).
    """
    interior = range(1, len(xlist)-1)   # interior node indices
    F_interior = sp.zeros(len(interior), 1)

    n2 = 5
    xi, ci = gl.gauleg(n2)

    def phi_i(i, x):
        return phi(i, x, xlist)  # your standard piecewise "hat" function

    # local routine to integrate f(x)*phi_i(x) over [a,b], 
    # handling the 1/3 discontinuity
    def piecewise_integral(a, b, i):
        # if entire [a,b] is below 1/3 or above 1/3, do one G-L integral
        discont = 0.1
        if b <= discont or a >= discont:
            return gauss_legendre_integrate(a, b, i) 
        # otherwise split at x=1/3
        part1 = gauss_legendre_integrate(a, discont, i)
        part2 = gauss_legendre_integrate(discont, b, i)
        return part1 + part2

    # single G-L integral on [A,B]
    def gauss_legendre_integrate(A, B, i):
        ssum = 0
        for k in range(n2):
            # map xi[k] ∈ [-1,1] → [A,B]
            x_mapped = 0.5*((B - A)*xi[k] + (A + B))
            w = ci[k]
            # integrand = f(x)*phi_i(i,x)
            integr_val = f(x_mapped)*phi_i(i, x_mapped)
            ssum += w*integr_val
        return 0.5*(B - A)*ssum

    # Fill each interior row
    for local_i, global_i in enumerate(interior):
        total = 0.0
        # sum over all mesh elements
        for e in range(len(xlist)-1):
            a = xlist[e]
            b = xlist[e+1]
            # integrate f(x)*phi_{global_i}(x) over [a,b], splitting if needed
            val = piecewise_integral(a, b, global_i)
            total += val
        F_interior[local_i,0] = total

    return F_interior

def C(matrixS, matrixF):
    a = matrixS.inv()
    return a * matrixF 

def approx_new(c, F):
    approx_new = c.T *F

    return approx_new[0,0]


def energy_norm(approximation):
    actual = 47/81
    energy_norm = actual - approximation
    return energy_norm

def solve_scF_once(n_list, m_list, mesh):
    """
    Build and solve the system S*C = F for a single iteration.

    Parameters
    ----------
    n_list, m_list : lists of indices (e.g. polynomial degrees)
    i_list         : indices for the piecewise-constant or piecewise-linear basis

    Returns
    -------
    C : Sympy Matrix
        The solution vector for the unknowns.
    """

    # Build the big Kronecker products
    matrixS1_Amn = A_S1(matrixA=A_matrix(n_list,m_list), matrixS1=S1(func= a1, mesh = mesh))          # = kronecker(A, S1_mat)
    matrixS0_delta = S0_delta(matrixS0=S0(func = a0, mesh = mesh), matrixdelta=delta(n_list,m_list))
    S = S_in_jm(matrixS1_Amn, matrixS0_delta) # S = matrixS0_delta + matrixS1_Amn

    # 2) Build the right-hand side vector F
    Fvec = build_force_vector2(num_nodes=len(mesh), xlist = mesh, f = f)

    # 3) Solve S*C = F for C
    C_sol = C(S, Fvec)  # calls your function C(...) which does: S_inv = S.inv(); C = S_inv*F


    return C_sol, Fvec

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
n = [0,1]
m = [0,1]
c_sol =solve_scF_once(n_list=n, m_list=m, mesh = gl.listi(0, 1, 2 **(-6), 2**6 + 1))[0]
nodal = assemble_nodal_values(c_sol)
x_nodal = np.array(gl.listi(0, 1, 2 **(-6), 2**6 + 1), dtype=float)
u_nodal = np.array(nodal, dtype=float)

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