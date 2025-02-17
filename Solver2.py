import gauleg as gl 
import sympy as sp 
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt

#####################################
###### INITIALISING PARAMETERS ######
#####################################

# def a0(x):
#     return 2 + math.sin(2*math.pi *x)

# def a0(x):
#     if 0 <= x < xd:
#         return 1
#     elif 0.5 <= x <= 1:
#         return 2

import numpy as np

def a0(x, beta):
    """
    Computes the coefficient a0 at points x using the parameter beta.
    
    Parameters:
        x (array-like): Spatial coordinates.
        beta (tuple): (x_star, r) where x_star is the center of the interval 
                      and r is the half-width of the interval.
    
    Returns:
        np.array: Coefficient values; 2 if x is inside the interval, 1 otherwise.
    """
    x_star, r = beta
    x = np.asarray(x)
    return np.where((x >= (x_star - r)) & (x <= (x_star + r)), 2.0, 1.0)


def f(x, beta):
    x = np.asarray(x)
    return a0(x, beta) * np.pi**2 * np.sin(np.pi * x)

n2 =5

####################################
######### HELPER FUNCTIONS #########
####################################

    
# helper functions
def GL(x_left, x_right, func):
    # Ensure xi and ci are NumPy arrays:
    xi, ci = np.polynomial.legendre.leggauss(n2)
    # Map the nodes from [-1,1] to [x_left, x_right]
    x_mapped = 0.5 * ((x_right - x_left) * xi + (x_right + x_left))
    # Evaluate the integrand at all mapped nodes (func must be vectorized or accept an array)
    integrand_values = func(x_mapped)
    # Compute the weighted sum using a dot-product and return the scaled result.
    return 0.5 * (x_right - x_left) * np.dot(ci, integrand_values)

def piecewise_GL(integrand, x_left, x_right, discont_points=None):
    """
    Integrate 'integrand(x)' from x_left to x_right using Gauss-Legendre quadrature,
    splitting the integration at any discontinuity points provided in discont_points.
    
    Parameters:
      integrand      : function to integrate, which must accept a NumPy array.
      x_left, x_right: the endpoints of the integration interval.
      n2             : number of Gauss–Legendre quadrature points.
      discont_points : a list (or scalar) of discontinuity points in (x_left, x_right).
                       If None or empty, no splitting is performed.
    
    Returns:
      The value of the integral.
    """
    # If no discontinuity is provided, do a single integration.
    if discont_points is None or len(discont_points) == 0:
        return GL(x_left=x_left, x_right=x_right, func=integrand)
    
    # Ensure discont_points is a list; if it's a scalar, convert it.
    if not isinstance(discont_points, (list, tuple, np.ndarray)):
        discont_points = [discont_points]
    
    # Filter out those discontinuity points that lie within (x_left, x_right)
    splits = [p for p in discont_points if x_left < p < x_right]
    
    # Build the list of subinterval endpoints
    pts = [x_left] + sorted(splits) + [x_right]
    
    # Integrate over each subinterval and sum the results.
    total = 0.0
    for i in range(len(pts) - 1):
        total += GL(x_left=pts[i], x_right=pts[i+1], func=integrand)
    return total

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
def phi_i(i, x, mesh):
    """
    Standard 1D hat (finite element) function.
    This returns the value of the i-th hat function at x, given a mesh.
    """
    x = np.asarray(x)  # Ensure x is a NumPy array.
    N = len(mesh) - 1  # Number of elements; nodes are 0,1,...,N.
    
    if i == 0:
        # For the left boundary, phi_0 is nonzero on [mesh[0], mesh[1]]
        cond = (x >= mesh[0]) & (x <= mesh[1])
        return np.where(cond, (mesh[1] - x) / (mesh[1] - mesh[0]), 0.0)
    
    elif i == N:
        # For the right boundary, phi_N is nonzero on [mesh[N-1], mesh[N]]
        cond = (x >= mesh[N-1]) & (x <= mesh[N])
        return np.where(cond, (x - mesh[N-1]) / (mesh[N] - mesh[N-1]), 0.0)
    
    else:
        # For an interior node i, the support is [mesh[i-1], mesh[i+1]]
        cond = (x >= mesh[i-1]) & (x <= mesh[i+1])
        # On the left subinterval [mesh[i-1], mesh[i]]
        left_cond = (x >= mesh[i-1]) & (x <= mesh[i])
        left_val = (x - mesh[i-1]) / (mesh[i] - mesh[i-1])
        # On the right subinterval (mesh[i], mesh[i+1]]
        right_cond = (x > mesh[i]) & (x <= mesh[i+1])
        right_val = (mesh[i+1] - x) / (mesh[i+1] - mesh[i])
        # Combine the two pieces:
        val = np.where(left_cond, left_val, np.where(right_cond, right_val, 0.0))
        return np.where(cond, val, 0.0)


def assemble_nodal_values(C):
    C = np.asarray(C)  # Ensure C is a NumPy array.
    return np.concatenate(([[0.0]], C, [[0.0]]))

def get_discont_points(x_left, x_right, beta):
    """
    Compute the discontinuity points for the coefficient a0(x, beta) on the interval [x_left, x_right].

    Parameters:
      x_left, x_right : float
          Endpoints of the integration interval.
      beta : tuple
          (x_star, r), where the discontinuity endpoints are x_star - r and x_star + r.
    
    Returns:
      A list of discontinuity points (subset of [x_star - r, x_star + r]) that lie in (x_left, x_right).
    """
    x_star, r_val = beta
    pts = []
    p1 = x_star - r_val
    p2 = x_star + r_val
    if x_left < p1 < x_right:
        pts.append(p1)
    if x_left < p2 < x_right:
        pts.append(p2)
    return pts


def solve_scF_once(mesh, beta):
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
    a0_with_beta = lambda x: a0(x, beta)

    S0_mat = S0_ji(a0_with_beta, mesh, beta)         # Possibly a Sympy matrix
    fvect = build_force_vector(f, mesh, 5, beta)  # Possibly a Sympy matrix

    
    # Extract interior (numerical slicing)
    S_int = S0_mat[1:-1, 1:-1]
    F_int = fvect[1:-1, :]
    c_sol = np.linalg.solve(S_int.T, F_int) 

    return c_sol, fvect

def refinement_loop(epsilon, beta):
    """
    1) Start with initial mesh
    2) Solve once
    3) Estimate errors
    4) If all errors < epsilon, done. Else refine, go back to step 2.
    """
    mesh = np.linspace(0.0, 1.0, 8).tolist()

    # Keep track of (mesh, energy_norm) in each iteration
    iteration_index = 0
    while True:
        # Solve for c_sol on the current mesh
        c_sol, f_sol = solve_scF_once(mesh=mesh, beta = beta)

        # Convert solution to nodal representation
        nodal = assemble_nodal_values(c_sol)

        # Estimate the elementwise errors
        errors = sum_of_error_list(mesh=mesh, nodal=nodal, beta= beta)

        # Mark which elements to refine
        elements_to_refine = element_selection(errors=errors, epsilon=epsilon)
        #elements_to_refine = dorfler_marking(errors, 0.9)

        # If no elements exceed threshold => done
        if not elements_to_refine:
            break

        # Refine the mesh only on the refinable elements.
        new_mesh = element_refinement(mesh, elements_to_refine)
        if new_mesh == mesh:
            print("Mesh did not change upon refinement. Terminating.")
            break

        mesh = new_mesh
        iteration_index += 1
        

    # After loop, final solution is c_sol on final mesh
    # Return everything, including the entire history
    return mesh, c_sol


###################################
####### ASSEMBLY OF MATRIX  #######
###################################

# assembly of matrix S0 and F
def S0_ji(func, mesh, beta):
    """
    Assemble the stiffness matrix S0 using the coefficient function 'func' (which is beta-aware)
    and the mesh. The integration splits at the discontinuity points derived from beta.
    """
    N = len(mesh)
    S0_mat = np.zeros((N, N), dtype=float)

    # Assembly of diagonal entries:
    for j in range(N):
        diag_val = 0.0
        # Contribution from the left subinterval [mesh[j-1], mesh[j]]
        if j > 0:
            x_left = mesh[j-1]
            x_right = mesh[j]
            def integrand_left(x):
                return func(x) * (dphi_i_on_element(j, j-1, mesh))**2
            diag_val += piecewise_GL(integrand_left, x_left, x_right,
                                      discont_points=get_discont_points(x_left, x_right, beta))
        # Contribution from the right subinterval [mesh[j], mesh[j+1]]
        if j < N-1:
            x_left = mesh[j]
            x_right = mesh[j+1]
            def integrand_right(x):
                return func(x) * (dphi_i_on_element(j, j, mesh))**2
            diag_val += piecewise_GL(integrand_right, x_left, x_right,
                                      discont_points=get_discont_points(x_left, x_right, beta))
        S0_mat[j, j] = diag_val

    # Assembly of off-diagonal entries:
    for j in range(1, N):
        x_left = mesh[j-1]
        x_right = mesh[j]
        def integrand_off(x):
            return func(x) * dphi_i_on_element(j-1, j-1, mesh) * dphi_i_on_element(j, j-1, mesh)
        val = piecewise_GL(integrand_off, x_left, x_right,
                           discont_points=get_discont_points(x_left, x_right, beta))
        S0_mat[j, j-1] = val
        S0_mat[j-1, j] = val  # Exploiting symmetry
    
    return S0_mat


def build_force_vector(f, mesh, n2=5, beta=None):
    """
    Assemble the force (load) vector F where
         F[i] = ∫ f(x, beta) * phi_i(x, mesh) dx,
    splitting the integration at the discontinuity points derived from beta.
    
    Parameters:
      f     : The source function, which now depends on beta.
      mesh  : List of node coordinates.
      n2    : Number of Gauss-Legendre points (if used in GL).
      beta  : Parameter tuple (x_star, r) used in f.
      
    Returns:
      F : A column vector (Sympy Matrix) of size (N+1) x 1.
    """
    num_nodes = len(mesh)
    F = np.zeros((num_nodes, 1), dtype=float)

    
    # Loop over each finite element function phi_i.
    for i in range(num_nodes):
        total = 0.0
        # Left subinterval (if it exists)
        if i > 0:
            x_left  = mesh[i-1]
            x_right = mesh[i]
            def integrand_left(x):
                return f(x, beta) * phi_i(i, x, mesh)
            total += piecewise_GL(integrand_left, x_left, x_right,
                                  discont_points=get_discont_points(x_left, x_right, beta))
        # Right subinterval (if it exists)
        if i < num_nodes - 1:
            x_left  = mesh[i]
            x_right = mesh[i+1]
            def integrand_right(x):
                return f(x, beta) * phi_i(i, x, mesh)
            total += piecewise_GL(integrand_right, x_left, x_right,
                                  discont_points=get_discont_points(x_left, x_right, beta))
        
        F[i, 0] = total
    return F


##############################################
####### ERROR INDICATOR AND REFINEMENT #######
##############################################


# def r(mesh, e):
#     """
#     Compute an approximation to the cell residual on element T = [mesh[e], mesh[e+1]].
#     Since u_h is piecewise linear and a0 is constant on T (if T does not cross x=1/3),
#     the derivative term is zero and we approximate r(x) by f(x).
#     """
#     a, b = mesh[e], mesh[e+1]
#     # Use a vectorized quadrature routine to approximate the L2 norm of f over T.
#     rT = GL(a, b, f) / (b - a)
#     return rT

def r(x, mesh, uh, beta):
    # 'uh' and 'mesh' are not strictly needed since a'(x)=0 almost everywhere.
    # The interior PDE says r(x) = f(x) + slope*a'(x), and here r(x) is taken as f(x).
    return f(x, beta)

def element_residual_l2(mesh, e, beta):
    """
    Compute the L2 norm squared of the cell residual on element T = [mesh[e], mesh[e+1]].
    If r(x) is piecewise constant on T, then the L2 norm squared is (r_T)^2 * h.
    When T is cut by a discontinuity, we integrate piecewise.
    """
    a, b = mesh[e], mesh[e+1]
    h = b - a
    
    discont_points = get_discont_points(a, b, beta)
    
    # Define the integrand that includes the dependence on beta.
    def interior_integrand(x_val):
        # Use r(x, ..., beta) so that the effect of beta is incorporated.
        return r(x_val, None, None, beta)**2

    # Integrate using piecewise_GL, splitting at the discontinuities if needed.
    r_sq = piecewise_GL(interior_integrand, x_left=a, x_right=b, discont_points=discont_points)
    return r_sq


def slope_at_node(mesh, uh, i, side):
    """
    Return the slope of the piecewise-linear FE solution uh on the element
    adjacent to node i from the given side ('left' or 'right').
    """
    n = len(mesh) - 1
    if side == 'left':
        if i == 0:
            return 0.0
        else:
            dx = mesh[i] - mesh[i-1]
            return (uh[i] - uh[i-1]) / dx
    elif side == 'right':
        if i >= n:
            return 0.0
        else:
            dx = mesh[i+1] - mesh[i]
            return (uh[i+1] - uh[i]) / dx
    else:
        raise ValueError("side must be 'left' or 'right'.")

def flux_jump(mesh, uh, i, a0_func):
    """
    Compute the jump in the numerical flux at the node mesh[i] (assumed interior).
    The flux is sigma = a0 * (approximate derivative). We define:
      sigma_left  = a0(x_i^-) * slope on [mesh[i-1], mesh[i]],
      sigma_right = a0(x_i^+) * slope on [mesh[i], mesh[i+1]].
    The jump is then: j(x_i) = sigma_right - sigma_left.
    
    Parameters:
      mesh    : array of node coordinates.
      uh      : array of nodal values of the FE solution.
      i       : the index of an interior node.
      a0_func : a function to evaluate a0 at a given x.
    
    Returns:
      float: the flux jump at node i.
    """
    slope_left = slope_at_node(mesh, uh, i, 'left')
    slope_right = slope_at_node(mesh, uh, i, 'right')
    # Use a small perturbation to evaluate a0 on either side.
    a_left  = a0_func(mesh[i] - 1e-9)
    a_right = a0_func(mesh[i] + 1e-9)
    sigma_left  = a_left * slope_left
    sigma_right = a_right * slope_right
    return sigma_right - sigma_left

def sum_of_error(i, mesh, nodal, beta):
    """
    Compute the error indicator for element i.
    It consists of two parts:
      - A residual term: h^2 * (L2 norm squared of the residual on element i).
      - A boundary term: h * (flux jump at the element boundary)^2.
    """
    x_left = mesh[i]
    x_right = mesh[i+1]
    h = x_right - x_left
    # Residual error on the element (incorporating beta)
    residual_sq = element_residual_l2(mesh, i, beta)
    # Flux jump error. Note: pass a lambda that fixes beta in the coefficient function.
    boundary_sq = flux_jump(mesh, nodal, i, lambda x: a0(x, beta)) ** 2
    return h**2 * residual_sq + h * boundary_sq


def sum_of_error_list(mesh, nodal, beta):
    """
    Return the error indicator for each element, given the mesh, nodal values, and beta.
    """
    return [sum_of_error(i, mesh, nodal, beta) for i in range(len(mesh) - 1)]


def refine_mesh(mesh, element_index):
    """
    Refine the element [mesh[element_index], mesh[element_index+1]] by bisection.
    """
    x_left = mesh[element_index]
    x_right = mesh[element_index+1]
    midpoint = 0.5 * (x_left + x_right)
    # Insert the midpoint after mesh[element_index]
    return mesh[:element_index+1] + [midpoint] + mesh[element_index+1:]


def element_selection(errors, epsilon):
    """
    Given an array-like 'errors' (one error per element) and a tolerance epsilon,
    return a list of element indices to refine (sorted in descending order).
    """
    errors = np.asarray(errors)
    # Find all indices where error exceeds epsilon.
    indices = np.nonzero(errors > epsilon)[0]
    # Sort in descending order so that when refining, index shifts are avoided.
    indices = np.sort(indices)[::-1]
    return indices.tolist()

def element_refinement(mesh, element_indices):
    mesh_arr = np.array(mesh)
    element_indices = np.array(element_indices, dtype=int)
    
    # Compute midpoints for each marked element.
    midpoints = 0.5 * (mesh_arr[element_indices] + mesh_arr[element_indices + 1])
    
    # Concatenate the original mesh with the new midpoints, then sort.
    new_mesh = np.sort(np.concatenate((mesh_arr, midpoints)))
    return new_mesh.tolist()




import numpy as np
import matplotlib.pyplot as plt

beta = np.array([0.5, 1/6])
mesh, c_sol= refinement_loop(epsilon=0.0001, beta= beta)


nodal = assemble_nodal_values(c_sol)  

x_nodal = np.array(mesh, dtype=float)
u_nodal = np.array(nodal, dtype=float)

# Define the exact solution
def exact_solution(x):
    x = np.asarray(x)
    return np.sin(np.pi*x)

# We'll plot the exact solution on a fine grid from 0..1
x_fine = np.linspace(0.0, 1.0, 200)
u_exact = exact_solution(x_fine)

plt.figure(figsize=(16,12))

# 1) Plot the piecewise-linear solution
plt.plot(
    x_nodal,
    u_nodal,
    marker='x',
    linestyle='-',
    color='blue',
    label='Numerical (Refined Mesh)'
)

# # 2) Plot the exact solution as a smooth curve
# plt.plot(
#     x_fine,
#     u_exact,
#     color='red',
#     linewidth=2,
#     label='Exact: sin(pi x)'
# )
plt.axvline(x=1/3, color='r', linestyle='--', linewidth=2, label="x = 1/3")
plt.axvline(x=2/3, color='r', linestyle='--', linewidth=2, label="x = 2/3")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Refined Numerical Solution vs. Exact')
plt.grid(True)
plt.legend()
plt.show()
