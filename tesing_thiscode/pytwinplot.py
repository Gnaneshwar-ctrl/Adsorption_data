import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def compute_r2_and_plot(true_data_file, algo1_file, algo2_file, algo3_file):
    """
    Compute R² for each column between the true data (true_data_file) 
    and three algorithms' simulated data (algo1_file, algo2_file, algo3_file). 
    Then, plot horizontal bar charts for the R² values, sorted in ascending order.
    """
    # Read the CSV files into dataframes
    true_data = pd.read_csv(true_data_file)
    mean = true_data.mean()
    std = true_data.std()
    
    algo1_data = pd.read_csv(algo1_file)
    algo2_data = pd.read_csv(algo2_file)
    algo3_data = pd.read_csv(algo3_file)
    
    # Normalize algo3 data using the mean and std of the true data
    algo3_data = algo3_data * std + mean

    # Ensure the data is read as DataFrames
    if isinstance(true_data, pd.DataFrame) and isinstance(algo1_data, pd.DataFrame) and isinstance(algo2_data, pd.DataFrame) and isinstance(algo3_data, pd.DataFrame):
        # Proceed with R² calculation and sorting
        r2_algo1 = []
        r2_algo2 = []
        r2_algo3 = []
        column_names = true_data.columns

        for col in column_names:
            true_col = true_data[col]
            algo1_col = algo1_data[col]
            algo2_col = algo2_data[col]
            algo3_col = algo3_data[col]

            # Calculate R² values
            r2_algo1.append(r2_score(true_col, algo1_col))
            r2_algo2.append(r2_score(true_col, algo2_col))
            r2_algo3.append(r2_score(true_col, algo3_col))

        # Create a DataFrame to store R² values
        r2_df = pd.DataFrame({
            'Column': column_names,
            'PPCA': r2_algo1,
            'AM': r2_algo2,
            'PKPCA': r2_algo3
        })

        # Sort the DataFrame by R² of PPCA in ascending order
        r2_df = r2_df.sort_values(by='PPCA', ascending=True)

        # Split the dataframe into two parts for better readability
        midpoint = len(r2_df) // 2
        r2_df1 = r2_df.iloc[:midpoint]
        r2_df2 = r2_df.iloc[midpoint:]

        # Plotting for the first half
        fig1, ax1 = plt.subplots(figsize=(10, len(r2_df1) * 0.3))
        bar_width = 0.25
        index1 = np.arange(len(r2_df1))

        ax1.barh(index1, r2_df1['PPCA'], bar_width, label='PPCA', color='b')
        ax1.barh(index1 + bar_width, r2_df1['AM'], bar_width, label='AM', color='g')
        ax1.barh(index1 + 2 * bar_width, r2_df1['PKPCA'], bar_width, label='PKPCA', color='r')

        ax1.set_xlabel('R² Value')
        ax1.set_title('R² Comparison of True Data with Simulated Data (Part 1)')
        ax1.set_yticks(index1 + bar_width / 2)
        ax1.set_yticklabels(r2_df1['Column'])
        ax1.legend()

        plt.tight_layout()
        plt.savefig('r2plot_part1.png', format='png', dpi=300)

        # Plotting for the second half
        fig2, ax2 = plt.subplots(figsize=(10, len(r2_df2) * 0.3))
        index2 = np.arange(len(r2_df2))

        ax2.barh(index2, r2_df2['PPCA'], bar_width, label='PPCA', color='b')
        ax2.barh(index2 + bar_width, r2_df2['AM'], bar_width, label='AM', color='g')
        ax2.barh(index2 + 2 * bar_width, r2_df2['PKPCA'], bar_width, label='PKPCA', color='r')

        ax2.set_xlabel('R² Value')
        ax2.set_title('R² Comparison of True Data with Simulated Data (Part 2)')
        ax2.set_yticks(index2 + bar_width / 2)
        ax2.set_yticklabels(r2_df2['Column'])
        ax2.legend()

        plt.tight_layout()
        plt.savefig('r2plot_part2.png', format='png', dpi=300)

    else:
        raise ValueError("One of the files is not properly read into a DataFrame.")


# Example usage
csv_file1 = 'truematrix.csv'
csv_file2 = 'imputedmatrix_0_PPCA_0.csv'
csv_file3 = 'imputedmatrix_0_AM_0.csv'
csv_file4 = 'X_reconstructed_rbf.csv'

compute_r2_and_plot(csv_file1, csv_file2, csv_file3, csv_file4)