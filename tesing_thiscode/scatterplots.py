import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('PCs.csv')

for p in range(2, 10):
    x_column = 1  
    y_column = p  


    plt.figure(figsize=(10, 6))
    plt.scatter(df[str(x_column)], df[str(y_column)], color='blue', alpha=0.5)

    for i in range(len(df)):
        plt.annotate(f'{i+1}', 
                    (df[str(x_column)][i], df[str(y_column)][i]), 
                    textcoords="offset points", 
                    xytext=(5,5), 
                    ha='center')


    plt.xlabel(f'PC {int(x_column)+1}')
    plt.ylabel(f'PC {int(y_column)+1}')
    plt.title(f'Scatter plot of PC {int(x_column)+1} vs PC {int(y_column)+1}')
    # plt.xlim(-0.00001, 0.00001)
    # plt.ylim(-0.00001, 0.00001)

    plt.grid(True)
    plt.show()