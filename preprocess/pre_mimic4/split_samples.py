import os
import pandas as pd
from pathlib import Path
from datetime import timedelta


p_project = Path(__file__).parents[2]
path_m4 = p_project/'data/mimic4'

# Path to the large CSV file
large_csv_path = path_m4/'processed/mimic4_full_dataset_gpts.csv'

# Create a directory to store the small CSV files

# delete the directory if it already exists and create a new one
os.system(f'rm -rf {path_m4}/processed/split')
os.system(f'mkdir {path_m4}/processed/split')

output_dir = path_m4/'processed/split'

# Read the large CSV file in chunks
chunk_size = 1000000  # Adjust the chunk size according to your memory capacity
reader = pd.read_csv(large_csv_path, chunksize=chunk_size, index_col=0)

# Iterate over each chunk
for i, chunk in enumerate(reader):

    # Iterate over each group
    for group_name in chunk.index.unique():
        group_data = chunk.loc[[group_name]]
        # Generate the output file path based on the group name
        output_file = os.path.join(output_dir, f'{group_name}.csv')
        
        # Save the group data to a small CSV file
        group_data.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
        
        print(f'Saved {output_file}')

print('Splitting complete.')
