import pandas as pd

# Load the CSV files
df1 = pd.read_csv('X_combined_linear_rbf.csv')
df2 = pd.read_csv('X_reconstructed_rbf.csv')

# Specify the columns you want to compare
column1 = 'o2_5bar'  # Update with the actual column name in file1
column2 = 'o2_5bar'  # Update with the actual column name in file2

# Extract the columns
data1 = df1[column1]
data2 = df2[column2]

# Ensure both columns have the same length
if len(data1) != len(data2):
    print("Warning: Columns are of different lengths. Comparisons will be truncated to the shortest length.")
    length = min(len(data1), len(data2))
    data1 = data1[:length]
    data2 = data2[:length]

# Compare the columns and count dissimilar cells
dissimilar_count = sum(data1 != data2)

print(f'Number of dissimilar cells: {dissimilar_count}')
