import math
import csv
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta


p_project = Path(__file__).parents[2]
path_m4 = p_project/'data/mimic4'


def normalize_csv(csv_file, fields):
    # Initialize dictionaries to store summation and count for each field
    field_sum = {field: 0 for field in fields}
    field_count = {field: 0 for field in fields}

    # First pass: Calculate summation and count for each field
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for field in fields:
                field_sum[field] += float(row["Value_" + field])
                field_count[field] += int(row["Mask_" + field])

    # Calculate mean for each field
    field_mean = {"Value_" + field: field_sum[field] /
                  field_count[field] for field in fields}

    print("Got mean")

    # Initialize dictionaries to store squared differences and count for each field
    field_squared_diff_sum = {field: 0 for field in fields}
    field_squared_diff_count = {field: 0 for field in fields}

    # Second pass: Calculate squared differences for each field
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for field in fields:
                if int(row["Mask_" + field]) == 1:
                    diff = float(row["Value_" + field]) - \
                        field_mean["Value_" + field]
                    field_squared_diff_sum[field] += diff ** 2
                    field_squared_diff_count[field] += 1

    # Calculate standard deviation for each field
    field_std_dev = {"Value_" + field: math.sqrt(field_squared_diff_sum[field] / field_squared_diff_count[field])
                     for field in fields}

    print("Got std")

    d_vm = {"Value_" + fd: "Mask_" + fd for fd in fields}
    # Third pass: Normalize values using mean and standard deviation
    # Write normalized rows to a new CSV file
    normalized_csv_file = csv_file.replace('.csv', '_normalized.csv')
    with open(csv_file, 'r') as f_in, open(normalized_csv_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            normalized_row = {}
            contains_outlier = False
            for field, value in row.items():
                mask = d_vm.get(field)
                if mask is not None and int(row[mask]) == 1:
                    normalized_value = (
                        float(value) - field_mean[field]) / field_std_dev[field]
                    if abs(normalized_value) > 5:
                        contains_outlier = True
                        break
                    normalized_row[field] = normalized_value
                else:
                    normalized_row[field] = value
            if not contains_outlier:
                writer.writerow(normalized_row)

    print(f"Normalized CSV file saved as {normalized_csv_file}")


# Example usage
csv_file = path_m4/'processed/mimic4_full_dataset_gpts.csv'

fields_to_normalize = []
for i in range(113):
    fields_to_normalize.append(str(i))

normalize_csv(str(csv_file), fields_to_normalize)


# Path to the large CSV file
large_csv_path = path_m4/'processed/mimic4_full_dataset_gpts_normalized.csv'

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
