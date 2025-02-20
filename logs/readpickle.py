import pickle 

filename = "FYP/logs/results_20250220_152021_f8926f7a.pkl"  # Replace with your actual file name
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Now you can access the loaded data, e.g.:
print(data)