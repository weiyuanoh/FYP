import numpy as np 
import matplotlib.pyplot as plt
import Solver3 as sl 

def interpolation_for_u_h(nodal, mesh, num_points):
  """
  Given the nodal solution 'nodal' computed on the refined mesh 'mesh',
  return the solution values at observation nodes computed as a uniform grid
  over [0,1] with num_points points.

  Assumes that every observation node exactly exists in the mesh.

  Parameters:
    nodal     : array-like, the computed solution at each mesh node.
    mesh      : array-like, the coordinates of the mesh nodes.
    num_points: int, number of observation points (e.g., 8).

  Returns:
    obs_values: np.array, the solution values corresponding to each observation node.

  Raises:
    ValueError if any observation node is not found in the mesh.
  """
  nodal = sl.assemble_nodal_values(nodal)
  mesh = np.asarray(mesh)
  nodal = np.asarray(nodal)
  
  # Compute observation nodes uniformly over [0,1]
  obs_nodes = np.linspace(0.0, 1.0, num_points)
  obs_values = np.zeros(num_points)

  for i, x_obs in enumerate(obs_nodes):
    idx = np.where(mesh == x_obs)[0]
    if idx.size == 0:
      raise ValueError(f"Observation node {x_obs} not found in the mesh.")
    obs_values[i] = nodal[idx[0]]

  return obs_values


def cov_matrix(sigma, num_points):
  sigma = (sigma ** 2) * np.eye(num_points)
  return sigma

def add_noise(observations_at_xi, num_points, sigma):
    """
    Adds a normally distributed noise, theta
    to observations from the forward solver.

    Arguments:
    observations_at_xi : observations at predetermined xi using interplotion. 
    num_points : how big your covariance matrix is 

    Returns:
    Delta : Array of Noisy observations.
    
    """
    sigma = cov_matrix(sigma, num_points)
    noise = np.random.multivariate_normal(np.zeros(num_points), sigma)
    delta = observations_at_xi + noise 
    return delta

def phi(observations, predicted, sigma, num_points) :
    '''
    For a set of predetermined points xi -- obtained via np.linspace,
    this function defines the likelihood function 

    Arguments:
    observations: Generated noisy observation using beta_true -- corresponds to y in literature
    predicted: For a proposed beta_i, we compute the noisy observation using the forward solver 
    -- corresponds to g(beta_i) in literature

    Returns: 
    Likelihood function that is proportional to the prior distribution
    
    '''
    covariance_matrix = cov_matrix(sigma, num_points)
    diff = predicted - observations
    covariance_matrix_inv = np.linalg.inv(covariance_matrix) # this supposed to be identity
    val = 0.5 * diff.T @ covariance_matrix_inv @ diff
    return val

def compute_A(phi_0, phi_i, sigma):

    val = np.exp(phi_0 - phi_i)
    
    return val

def MCMC(beta_true, number_of_iter, burn_in, sigma, num_points): 
    '''
    Builds a Markov Chain 

    Key Steps:
    1. Initialise a choice of Beta, beta_0 
    2. Compute likelihood of beta_0, using delta and beta_0_predicted
    3. Initialise the loop.
        - we propose a new beta_i from x* ~ Uniform(0.15, 0.85) and r ~ Uniform(0, 0.15)
        - compute y_i and g(beta_i)
        - compute likelihood using {y_i and g(beta_i)}
        - set alpha = min{1, likelihood }
    '''
    # set seed 
    np.random.seed(42)
    # range of uniform distribution 
    x_star_range = (0.3, 0.7) 
    r_range = (0.1, 0.2) 
    chain = []
    # compute delta 
    mesh_true , c_sol_true = sl.refinement_loop(0.0001, beta_true) 
    y_true = interpolation_for_u_h(c_sol_true, mesh_true, num_points)
    delta = add_noise(y_true, num_points, sigma)

    # draw first copy of beta --> beta_0

    beta_0 = np.array([
            np.random.uniform(*x_star_range),
            np.random.uniform(*r_range)
        ])
    print("Beta_0:", beta_0)
    
    # initialise current observations and likelihood 
    mesh_0, c_sol_0 = sl.refinement_loop(0.01, beta = beta_0)
    y_0 = interpolation_for_u_h(c_sol_0, mesh_0, num_points)
    phi_0 = phi(delta, y_0, sigma, num_points)
    print("phi_0:", phi_0)
    
    iter_count = 0
    acceptance_count = 0 
    acceptance_prob_history = []

    for i in range(number_of_iter):
        beta_proposal = np.array([
            np.random.uniform(*x_star_range),
            np.random.uniform(*r_range)
        ])
        print("beta proposal:", beta_proposal)
        mesh_proposal, c_sol_proposal = sl.refinement_loop(0.01, beta = beta_proposal)
        y_proposal = interpolation_for_u_h(c_sol_proposal, mesh_proposal, num_points)
        phi_proposal = phi(delta, y_proposal, sigma, num_points)
        print("phi proposal:", phi_proposal)
        # compute acceptance probability 
        A = compute_A(phi_0, phi_proposal, sigma)
        acceptance_prob = min(1, A)
        acceptance_prob_history.append(acceptance_prob)
        print("acceptance probablity:", acceptance_prob)
        
        # Accept or reject the proposal
        if np.random.rand() < acceptance_prob:
            beta_0 = beta_proposal # update the current state as the last accepted proposal
            y_0 = y_proposal # update the current observations to the last accepted observation
            phi_0 = phi_proposal
            acceptance_count += 1
        
        # Record the current state.
        chain.append(beta_0.copy())
        print("Chain length:", len(chain))
    
    chain = np.array(chain)
    # Compute the MCMC estimate as the mean of the samples after burn-in.
    beta_mcmc = np.mean(chain[burn_in:], axis=0)
    
    return chain, beta_mcmc, acceptance_prob_history, acceptance_count


# number_of_iter = 100
# burn_in = 10
# sigma = 0.01
# num_points = 100
# beta_true = np.array([0.65, 0.15])

# chain, beta_mcmc, acceptance_history, acceptance_count = MCMC(beta_true, number_of_iter, burn_in, sigma, num_points)
# print("True beta:", beta_true )
# print("MCMC estimated beta:", beta_mcmc)
# print("Acceptance Probability History:", acceptance_history)
# print("Acceptance Count:", acceptance_count)