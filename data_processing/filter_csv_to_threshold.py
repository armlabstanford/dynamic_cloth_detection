import pandas as pd
import os

def filter_data(input_file, output_file, threshold=-1.25):
    # Load the CSV file
    data = pd.read_csv(input_file)
    
    # Find the first occurrence where force_x exceeds the threshold
    index_threshold = data[data['force_x'] > threshold].index[0]
    
    # Keep all rows after this index
    filtered_data = data.iloc[index_threshold:]
    
    # Save the filtered data to a new CSV file
    filtered_data.to_csv(output_file, index=False)

def process_files(input_dir, output_dir, prefix='wrench_data_', threshold=-1.25):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    trial_num = 20
    while True:
        input_file = os.path.join(input_dir, f'{prefix}{trial_num}.csv')
        if not os.path.exists(input_file):
            break
        output_file = os.path.join(output_dir, f'filtered_{prefix}{trial_num}.csv')
        filter_data(input_file, output_file, threshold)
        print(f'Processed {input_file} -> {output_file}')
        trial_num += 1

# Example usage
#input_dir = 'wrench_data' # change for each trial num
layer_str = 'two_layers'
wrench_dir = f'tests_5_23/wrench_data/{layer_str}'
output_dir = f'output_tests_5_23/wrench_data/{layer_str}'
process_files(wrench_dir, output_dir)
