import pickle 

filename = r"FYP\logs\results_20250220_183435_beta=0.65-0.15_iter=5000_burn=2000_sigma=0.01_points=200_ab93e248.pkl"  # Replace with your actual file name
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Now you can access the loaded data, e.g.:
print(data)