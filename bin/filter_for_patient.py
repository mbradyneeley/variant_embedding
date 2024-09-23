import pandas as pd

# Input and output file paths
input_csv = '~/data/basic_AF_0.05_filtered_BG_train.csv'  # Replace with your large CSV file path
output_csv = './filtered_data.csv'

# Identifier to filter
identifier_to_filter = 709426

# Initialize an empty DataFrame to store filtered rows
filtered_data = pd.DataFrame()

# Use chunksize to handle large file
chunksize = 10**6  # Adjust based on available memory

# Reading and filtering the large CSV in chunks
for chunk in pd.read_csv(input_csv, chunksize=chunksize):
    # Filter by the 'identifier' column
    filtered_chunk = chunk[chunk['identifier'] == identifier_to_filter]
    
    # Append the filtered data to the final DataFrame
    filtered_data = pd.concat([filtered_data, filtered_chunk])

# Write the filtered data to a new CSV file
filtered_data.to_csv(output_csv, index=False)

print(f'Filtered data saved to {output_csv}')

