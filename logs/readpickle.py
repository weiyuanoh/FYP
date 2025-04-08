import pickle 
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

filename = r"FYP\logs\results_20250221_112433_beta_true=0.65-0.15_iter=10000_burn=5000_sigma=0.01_points=100_cde886ab.pkl"  # Replace with your actual file name
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Now you can access the loaded data, e.g.:
# 2. Extract the chain (shape: [num_samples, num_params])
chain = data['results']['chain']

# 3. Plot histograms of each parameter
#    Here, chain[:, 0] is the first parameter, chain[:, 1] is the second, etc.
plt.hist(chain[:, 0], bins=50)
plt.title("Distribution of beta[0]")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(chain[:, 1], bins=50)
plt.title("Distribution of beta[1]")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


chain_burned = chain[5000:, :]

num_samples = len(chain_burned)  # or chain.shape[0]

# Create the iteration index
iterations = np.arange(num_samples)


# Repeat for beta[1] if you like
plt.scatter(iterations, chain_burned[:, 1], s=2)
plt.title("Scatter Plot of beta[1]")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show()
