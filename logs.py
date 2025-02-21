import argparse
import logging
import uuid
import pickle
from datetime import datetime
import os

import numpy as np
import Solver3 as sl
import Bayesian_Inverse as bi



LOG_DIR = r"C:\Users\Admin\PycharmProjects\FYP\.venv\FYP\logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'computation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_run_id(params):
    """
    Generate a custom run ID based on the parameters.
    
    Args:
        params (dict): A dictionary containing the run parameters.
        
    Returns:
        str: A custom run ID string.
    """
    # Convert beta_true list to a string, replacing decimals with dashes (or any other format you prefer)
    beta_str = "-".join(str(b) for b in params['beta_true'])
    
    # Create a custom run_id using timestamp and parameters.
    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"beta_true={beta_str}_"
        f"iter={params['number_of_iter']}_"
        f"burn={params['burn_in']}_"
        f"sigma={params['sigma']}_"
        f"points={params['num_points']}_"
        f"{str(uuid.uuid4())[:8]}"
    )
    return run_id
def run_MCMC(beta_true, number_of_iter, burn_in, sigma, num_points):
    chain, beta_mcmc, acceptance_count = bi.MCMC(beta_true, number_of_iter, burn_in, sigma, num_points)

    params = {
        'beta_true': beta_true,
        'number_of_iter': number_of_iter,
        'burn_in': burn_in,
        'sigma': sigma,
        'num_points': num_points
    }

    run_id = generate_run_id(params)
    logging.info(f"Starting MCMC with run_id={run_id}, params={params}")
    # Store the MCMC outputs inside `results`.
    results = {
        'chain': chain,
        'beta_mcmc': beta_mcmc,
        'acceptance_count': acceptance_count
    }

    result_dict = {
        'run_id': run_id,
        'params': params,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    filename = os.path.join(LOG_DIR, f"results_{run_id}.pkl")

    try:
        with open(filename, 'wb') as f:
            pickle.dump(result_dict, f)
        logging.info(f"Successfully saved MCMC results to {filename}")
    except Exception as e:
        logging.error(f"Error saving pickle file: {e}")

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Now you can access the loaded data, e.g.:
    print(data)

    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC and save results.")
    
    parser.add_argument("--beta_true", type=float, nargs="+", default=[0.65, 0.15],
                    help="Beta array (space separated floats).")
    parser.add_argument("--number_of_iter", type=int, default=10000,
                        help="Number of iterations for MCMC. Default=10000")
    parser.add_argument("--burn_in", type=int, default=1000,
                        help="Burn-in period. Default=1000")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Value of sigma for MCMC. Default=0.1")
    parser.add_argument("--num_points", type=int, default=100,
                        help="Number of data points. Default=100")


    args = parser.parse_args()

    # Now call run_MCMC with these arguments
    run_MCMC(args.beta_true, args.number_of_iter, args.burn_in, args.sigma, args.num_points)

