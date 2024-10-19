import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
true_data = pd.read_csv('X_combined_linear_rbf.csv')
generated_data = pd.read_csv('X_reconstructed_rbf.csv')

# Ensure both dataframes have the same columns in the same order
common_columns = true_data.columns.intersection(generated_data.columns)
true_data = true_data[common_columns]
generated_data = generated_data[common_columns]

# Plotting scatter plots for each corresponding column
for column in common_columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(true_data[column], generated_data[column], alpha=0.5, label='Generated vs True')
    plt.plot([true_data[column].min(), true_data[column].max()], [true_data[column].min(), true_data[column].max()], 'r--', label='True Prediction Line')
    plt.title(f'Scatter Plot for {column}')
    plt.xlabel('True Data')
    plt.ylabel('Generated Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot'+str(column)+'.png')
