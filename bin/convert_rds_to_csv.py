import pyreadr
import pandas as pd

# Define the file path for the RDS file
rds_file_path = '/mnt/atlas_local/guantong/home/Multi_agent/basic_AF_0.05_filtered_BG_train.rds'

# Read the RDS file
rds_data = pyreadr.read_r(rds_file_path)
print(rds_data)
#
## Extract the dataframe from the RDS file (assuming there's only one object in the file)
#df = list(rds_data.values())[0]
#
## Define the output CSV file path
#csv_file_path = '~/data/basic_AF_0.05_filtered_BG_train.csv'
#
## Write the dataframe to a CSV file
#df.to_csv(csv_file_path, index=False)
#
## Output for confirmation
#print(f"CSV file has been saved to: {csv_file_path}")
