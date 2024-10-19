import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def compute_mse_and_plot(true_data_file, algo1_file, algo2_file, algo3_file):
    """
    Compute MSE for each column between the true data (true_data_file) 
    and three algorithms' simulated data (algo1_file, algo2_file, algo3_file). 
    Then, plot horizontal bar charts for the MSE values, sorted in ascending order.
    """
    # Read the CSV files into dataframes
    true_data = pd.read_csv(true_data_file)
    
    algo1_data = pd.read_csv(algo1_file)
    algo2_data = pd.read_csv(algo2_file)
    algo3_data = pd.read_csv(algo3_file)
    
    # Normalize algo3 data using the mean and std of the true data
    mean = true_data.mean()
    std = true_data.std()
    algo3_data = algo3_data * std + mean

    # Ensure the data is read as DataFrames
    if isinstance(true_data, pd.DataFrame) and isinstance(algo1_data, pd.DataFrame) and isinstance(algo2_data, pd.DataFrame) and isinstance(algo3_data, pd.DataFrame):
        # Proceed with MSE calculation and sorting
        mse_algo1 = []
        mse_algo2 = []
        mse_algo3 = []
        column_names = true_data.columns

        for col in column_names:
            true_col = true_data[col]
            algo1_col = algo1_data[col]
            algo2_col = algo2_data[col]
            algo3_col = algo3_data[col]

            # Calculate MSE values
            mse_algo1.append(mean_squared_error(true_col, algo1_col))
            mse_algo2.append(mean_squared_error(true_col, algo2_col))
            mse_algo3.append(mean_squared_error(true_col, algo3_col))

        # Create a DataFrame to store MSE values
        mse_df = pd.DataFrame({
            'Column': column_names,
            'PPCA': mse_algo1,
            'AM': mse_algo2,
            'PKPCA': mse_algo3
        })

        # Sort the DataFrame by MSE of PPCA in ascending order
        mse_df = mse_df.sort_values(by='PPCA', ascending=True)

        # Split the dataframe into two parts for better readability
        midpoint = len(mse_df) // 2
        mse_df1 = mse_df.iloc[:midpoint]
        mse_df2 = mse_df.iloc[midpoint:]

        # Plotting for the first half
        fig1, ax1 = plt.subplots(figsize=(10, len(mse_df1) * 0.3))
        bar_width = 0.25
        index1 = np.arange(len(mse_df1))

        ax1.barh(index1, mse_df1['PPCA'], bar_width, label='PPCA', color='b')
        ax1.barh(index1 + bar_width, mse_df1['AM'], bar_width, label='AM', color='g')
        ax1.barh(index1 + 2 * bar_width, mse_df1['PKPCA'], bar_width, label='PKPCA', color='r')

        ax1.set_xlabel('MSE Value')
        ax1.set_title('MSE Comparison of True Data with Simulated Data (Part 1)')
        ax1.set_yticks(index1 + bar_width / 2)
        ax1.set_yticklabels(mse_df1['Column'])
        # ax1.set_xlim(0, 0.0000001)
        ax1.legend()

        plt.tight_layout()
        plt.savefig('mseplot_part1.png', format='png', dpi=300)

        # Plotting for the second half
        fig2, ax2 = plt.subplots(figsize=(10, len(mse_df2) * 0.3))
        index2 = np.arange(len(mse_df2))

        ax2.barh(index2, mse_df2['PPCA'], bar_width, label='PPCA', color='b')
        ax2.barh(index2 + bar_width, mse_df2['AM'], bar_width, label='AM', color='g')
        ax2.barh(index2 + 2 * bar_width, mse_df2['PKPCA'], bar_width, label='PKPCA', color='r')

        ax2.set_xlabel('MSE Value')
        ax2.set_title('MSE Comparison of True Data with Simulated Data (Part 2)')
        ax2.set_yticks(index2 + bar_width / 2)
        ax2.set_yticklabels(mse_df2['Column'])
        # ax2.set_xlim(0, 0.0000001)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('mseplot_part2.png', format='png', dpi=300)

    else:
        raise ValueError("One of the files is not properly read into a DataFrame.")


# Example usage
csv_file1 = 'truematrix.csv'
csv_file2 = 'imputedmatrix_0_PPCA_114.csv'
csv_file3 = 'imputedmatrix_0_AM_114.csv'
csv_file4 = 'X_reconstructed_rbf.csv'

compute_mse_and_plot(csv_file1, csv_file2, csv_file3, csv_file4)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error


# def compute_total_mse(true_data_file, algo1_file, algo2_file, algo3_file):
#     """
#     Compute the total MSE (sum of MSE values across all columns) 
#     between the true data (true_data_file) and three algorithms' 
#     simulated data (algo1_file, algo2_file, algo3_file).
    
#     Args:
#     - true_data_file: CSV file with the true data.
#     - algo1_file: CSV file with the first algorithm's simulated data.
#     - algo2_file: CSV file with the second algorithm's simulated data.
#     - algo3_file: CSV file with the third algorithm's simulated data.
    
#     Returns:
#     - Total MSE values for each algorithm.
#     """
#     # Read the CSV files into dataframes
#     true_data = pd.read_csv(true_data_file)
    
#     algo1_data = pd.read_csv(algo1_file)
#     algo2_data = pd.read_csv(algo2_file)
#     algo3_data = pd.read_csv(algo3_file)
    
#     # Normalize algo3 data using the mean and std of the true data
#     mean = true_data.mean()
#     std = true_data.std()
#     algo3_data = algo3_data * std + mean

#     # Ensure the data is read as DataFrames
#     if isinstance(true_data, pd.DataFrame) and isinstance(algo1_data, pd.DataFrame) and isinstance(algo2_data, pd.DataFrame) and isinstance(algo3_data, pd.DataFrame):
#         # Proceed with MSE calculation and summing
#         total_mse_algo1 = 0
#         total_mse_algo2 = 0
#         total_mse_algo3 = 0
#         column_names = true_data.columns

#         for col in column_names:
#             true_col = true_data[col]
#             algo1_col = algo1_data[col]
#             algo2_col = algo2_data[col]
#             algo3_col = algo3_data[col]

#             # Calculate and sum the MSE values
#             total_mse_algo1 += mean_squared_error(true_col, algo1_col)
#             total_mse_algo2 += mean_squared_error(true_col, algo2_col)
#             total_mse_algo3 += mean_squared_error(true_col, algo3_col)

#         # Print the total MSE values
#         print(f"Total MSE for PPCA: {total_mse_algo1}")
#         print(f"Total MSE for AM: {total_mse_algo2}")
#         print(f"Total MSE for PKPCA: {total_mse_algo3}")

#         return total_mse_algo1, total_mse_algo2, total_mse_algo3

#     else:
#         raise ValueError("One of the files is not properly read into a DataFrame.")


# # Example usage
# csv_file1 = 'truematrix.csv'
# csv_file2 = 'imputedmatrix_0_PPCA_114.csv'
# csv_file3 = 'imputedmatrix_0_AM_114.csv'
# csv_file4 = 'X_reconstructed_rbf.csv'

# compute_total_mse(csv_file1, csv_file2, csv_file3, csv_file4)