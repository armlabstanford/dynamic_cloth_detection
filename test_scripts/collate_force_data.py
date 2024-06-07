import numpy as np
import pandas as pd
import os

def load_force_data(directory):
    # Initialize an empty list to store data for all trials
    all_trials_data = []

    # Loop through all files in the directory
    for file in sorted(os.listdir(directory)):
        if file.endswith('.csv'):
            # Read the CSV file
            print(f"reading {file}")
            df = pd.read_csv(os.path.join(directory, file))
            
            # Extract the force data (skip the timestamp column)
            force_data = df.iloc[:, 1:].values

            force_data = np.linspace(force_data[0], force_data[-1], 23)
            
            # Append to the list of all trials data
            all_trials_data.append(force_data)

    # Convert the list to a NumPy array of shape (100, 23, 6)
    all_trials_data = np.array(all_trials_data)
    return all_trials_data

# Specify the directory containing the CSV files
directory = '/home/armlab/Documents/soft_manipulation/wrench_data/2_layer'

# Load the data
force_data_array = load_force_data(directory)

# Check the shape of the resulting NumPy array
print(force_data_array.shape)  # Should print (100, 23, 6)
np.savez('2layer_forces.npz', force_data_array=force_data_array)
