import math
import csv
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta


p_project = Path(__file__).parents[2]
path_m4 = p_project/'data/mimic4'


input_file = path_m4/'processed/mimic4_full_dataset_gpts_normalized.csv'
output_file = path_m4/'processed/mimic4_full_dataset_gpts_normalized_good.csv'

fields_to_check = []
for i in range(113):
    fields_to_check.append("Value_" + str(i))

with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
    reader = csv.DictReader(csv_input)
    writer = csv.DictWriter(csv_output, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        should_delete_row = False

        for field in fields_to_check:
            value = float(row[field])
            if abs(value) > 5:
                should_delete_row = True
                break

        if not should_delete_row:
            writer.writerow(row)

print("Rows with values greater than 5 in the specified fields have been deleted.")


# Path to the large CSV file
large_csv_path = path_m4/'processed/mimic4_full_dataset_gpts_normalized_good.csv'

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
        group_data.to_csv(output_file, index=False, mode='a',
                          header=not os.path.exists(output_file))

        print(f'Saved {output_file}')

print('Splitting complete.')


# Delete files with less than 10 data rows

folder_path = path_m4/'processed/split'  # Replace with the path to your folder

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Get the number of rows in the CSV file using shell command
    row_count_command = f"wc -l < {file_path}"
    row_count = int(os.popen(row_count_command).read().strip())

    # Check if the row count is less than 11 (data rows + header row)
    if row_count < 11:
        # Delete the file using shell command
        delete_command = f"rm {file_path}"
        os.system(delete_command)
        print(f"Deleted file: {file_path}")
