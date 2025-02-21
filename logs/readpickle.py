import pickle 

filename = r"FYP\logs\results_20250221_112433_beta_true=0.65-0.15_iter=10000_burn=5000_sigma=0.01_points=100_cde886ab.pkl"  # Replace with your actual file name
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Now you can access the loaded data, e.g.:
print(data)