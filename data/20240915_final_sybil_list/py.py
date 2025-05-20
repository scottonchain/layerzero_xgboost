import pandas as pd

# Load the CSV files
provisional_df = pd.read_csv('fcfs_provisional_list.csv', header=None)
final_df = pd.read_csv('fcfs_list.csv', header=None)

# Extract the first column (assumed to be addresses)
provisional_addresses = set(provisional_df.iloc[:, 0].str.lower())
final_addresses = set(final_df.iloc[:, 0].str.lower())

# Find addresses in provisional but not in final
missing_addresses = provisional_addresses - final_addresses

# Print the results
print(f"Missing addresses ({len(missing_addresses)} total):")
for addr in sorted(missing_addresses):
    print(addr)