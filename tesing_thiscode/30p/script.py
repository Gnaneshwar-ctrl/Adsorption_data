import numpy as np
from numpy.linalg import eig, inv, slogdet
from scipy.linalg import pinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

def kpca(kern, X, dims):
    A = kern_compute(kern, X)
    sigma, V, lambda_ = ppca(A, dims)
    W = V
    return W

def kpca_like_missing(X, W, sigma, kern, I, npts, Dim):
    X = X.reshape(npts, Dim)
    A = kern_compute(kern, X)
    Kx = W @ W.T + sigma * np.eye(npts)
    sign, logdet_Kx = slogdet(Kx)
    if sign <= 0:
        raise ValueError("Kx is not positive definite.")
    f = 0.5 * logdet_Kx + 0.5 * np.trace(A @ pinv(Kx)) - npts / 2
    return f

def kpca_like_missing_grad(X, W, sigma, kern, I, npts, Dim):
    X = X.reshape(npts, Dim)
    A = kern_compute(kern, X)
    Kx = W @ W.T + sigma * np.eye(npts)
    invKx = pinv(Kx)
    gX = np.zeros((npts, Dim))
    for i in range(npts):
        grad_kern = kern_grad_x(kern, X[i, :], X)
        gX[i, :] = (invKx[i, :] @ grad_kern) * I[i, :]
    f = gX.flatten()
    return f

def kpca_misser(X, p, seedling):
    if not 0 <= p <= 1:
        raise ValueError('p must be in [0,1]')
    
    np.random.seed(seedling)
    npts, Dim = X.shape
    
    # Calculate the total number of missing values
    total_missing = int(npts * Dim * p)
    
    # Create an array to track missing elements
    I = np.zeros((npts, Dim), dtype=float)
    
    # Randomly select indices to be missing
    missing_indices = np.random.choice(npts * Dim, total_missing, replace=False)
    
    for idx in missing_indices:
        row = idx // Dim
        col = idx % Dim
        I[row, col] = 1  # Mark this element as missing
    
    # Create the new matrix with missing values
    neuX = X * (1 - I)
    avgX = np.sum(neuX, axis=0) / np.sum(1 - I, axis=0, where=(1 - I) != 0)
    
    # Replace missing values with the average
    Replace = np.zeros((npts, Dim))
    for i in range(Dim):
        Replace[:, i] = avgX[i] * I[:, i]
    
    newX = neuX + Replace
    return newX, I

# def kpca_misser(X, p, seedling):
#     if not 0 <= p <= 1:
#         raise ValueError('p must be in [0,1]')
#     np.random.seed(seedling)
#     npts, Dim = X.shape
#     Y = np.random.rand(npts, Dim)
#     I = (Y > (1 - p)).astype(float)
#     neuX = X * (1 - I)
#     avgX = np.sum(neuX, axis=0) / np.sum(1 - I, axis=0)
#     Replace = np.zeros((npts, Dim))
#     for i in range(Dim):
#         Replace[:, i] = avgX[i] * I[:, i]
#     newX = neuX + Replace
#     return newX, I

def kpca_missing_data(X, options, W, sigma, kern, I, numComp):
    npts, Dim = X.shape
    diffKL = 1
    Iter = 0
    maxOuterIter = options['maxIter']
    sensitivity = options['sensitivity']
    
    while diffKL > sensitivity and Iter < maxOuterIter:
        Iter += 1
        X_flat = X.flatten()
        OldKL = kpca_like_missing(X_flat, W, sigma, kern, I, npts, Dim)
        
        # Define objective and gradient functions
        objective_values = []
        
        def objective(X_flat):
            value = kpca_like_missing(X_flat, W, sigma, kern, I, npts, Dim)
            objective_values.append(value)
            return value
        
        def gradient(X_flat):
            return kpca_like_missing_grad(X_flat, W, sigma, kern, I, npts, Dim)
        
        # Perform optimization
        res = minimize(
            objective, X_flat, jac=gradient, method='L-BFGS-B',
            options={'maxiter': 10000, 'gtol': 1e-5}
        )
        
        if not res.success:
            print('Optimization failed at iteration {}: {}'.format(Iter, res.message))
            break
        
        newX_flat = res.x
        NewKL = kpca_like_missing(newX_flat, W, sigma, kern, I, npts, Dim)
        diffKL = OldKL - NewKL
        
        if diffKL < 0:
            raise ValueError('KL divergence increased!')
        
        X = newX_flat.reshape(npts, Dim)
        A = kern_compute(kern, X)
        sigma, V, lambda_ = ppca(A, numComp)
        W = V @ np.sqrt(np.diag(lambda_))
        
        if options['display']:
            plt.figure()
            plt.plot(objective_values)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Function Value')
            plt.title('Objective Function over Iterations')
            plt.show()
        
        print(f'Iteration {Iter} complete. KL divergence decreased by {diffKL}')
    
    if Iter == maxOuterIter:
        print('Maximum iterations exceeded.')
    
    updatedX = X
    var = sigma
    PcEig = V
    PcCoeff = lambda_
    return updatedX, var, PcEig, PcCoeff

def kpca_num_comp(A, options):
    lambda_ = np.linalg.eigvalsh(A)
    total_variance = np.sum(lambda_)
    sorted_lambda = np.sort(lambda_)[::-1]
    cumulative_variance = np.cumsum(sorted_lambda) / total_variance
    numComp = np.searchsorted(cumulative_variance, options['captVar']) + 1
    return numComp

def kpca_options():
    options = {
        'maxIter': 25,
        'sensitivity': 1e-4,
        'display': True,
        'captVar': 0.95,
    }
    return options

def ppca(A, dims):
    lambda_, V = np.linalg.eigh(A)
    idx = np.argsort(lambda_)[::-1]
    lambda_ = lambda_[idx]
    V = V[:, idx]
    
    # Estimate sigma as the mean of the discarded eigenvalues
    if dims < len(lambda_):
        discarded_lambda = lambda_[dims:]
        sigma = np.mean(discarded_lambda)
    else:
        sigma = 1e-5
        print("Warning: dims >= number of eigenvalues. Setting sigma to a small positive value.")
    
    # Truncate to the desired number of dimensions
    lambda_ = lambda_[:dims]
    V = V[:, :dims]
    
    return sigma, V, lambda_

def kernel_function(x, y, kern):
    if kern['type'] == 'combined':
        combination = kern.get('combination', 'sum')
        kernels = kern['kernels']
        weights = kern.get('weights', [1]*len(kernels))
        if combination == 'sum':
            value = 0
            for k, w in zip(kernels, weights):
                value += w * kernel_function(x, y, k)
            return value
        elif combination == 'product':
            value = 1
            for k, w in zip(kernels, weights):
                value *= kernel_function(x, y, k) ** w
            return value
        else:
            raise ValueError("Unsupported combination method")
    elif kern['type'] == 'rbf':
        gamma = kern.get('gamma', 1.0)
        return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
    elif kern['type'] == 'polynomial':
        degree = kern.get('degree', 3)
        coef0 = kern.get('coef0', 1)
        alpha = kern.get('alpha', 1)
        return (alpha * np.dot(x, y) + coef0) ** degree
    elif kern['type'] == 'sigmoid':
        alpha = kern.get('alpha', 1.0)
        coef0 = kern.get('coef0', 0.0)
        return np.tanh(alpha * np.dot(x, y) + coef0)
    elif kern['type'] == 'cosine':
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0
        return np.dot(x, y) / (norm_x * norm_y)
    elif kern['type'] == 'linear':
        return np.dot(x, y)
    else:
        raise ValueError("Unsupported kernel type")

def kernel_gradient(x_i, X, kern):
    if kern['type'] == 'combined':
        combination = kern.get('combination', 'sum')
        kernels = kern['kernels']
        weights = kern.get('weights', [1]*len(kernels))
        if combination == 'sum':
            grad = np.zeros_like(X)
            for k, w in zip(kernels, weights):
                grad += w * kernel_gradient(x_i, X, k)
            return grad
        elif combination == 'product':
            n = X.shape[0]
            Dim = X.shape[1]
            grad = np.zeros((n, Dim))
            for i in range(len(kernels)):
                k = kernels[i]
                w = weights[i]
                # Compute the product of all kernels except the i-th
                product = np.ones(n)
                for j in range(len(kernels)):
                    if j != i:
                        product *= kernel_function(x_i, X, kernels[j]) ** weights[j]
                # Multiply by the gradient of the i-th kernel
                grad += w * product[:, np.newaxis] * kernel_gradient(x_i, X, k)
            return grad
        else:
            raise ValueError("Unsupported combination method")
    elif kern['type'] == 'rbf':
        gamma = kern.get('gamma', 1.0)
        diff = X - x_i
        K = np.exp(-gamma * np.sum(diff ** 2, axis=1))
        grad = 2 * gamma * diff * K[:, np.newaxis]
    elif kern['type'] == 'polynomial':
        degree = kern.get('degree', 3)
        coef0 = kern.get('coef0', 1)
        alpha = kern.get('alpha', 1)
        inner_product = alpha * X @ x_i + coef0
        K = inner_product ** (degree - 1)
        grad = degree * alpha * K[:, np.newaxis] * X
    elif kern['type'] == 'sigmoid':
        alpha = kern.get('alpha', 1.0)
        coef0 = kern.get('coef0', 0.0)
        inner_product = alpha * X @ x_i + coef0
        K = np.tanh(inner_product)
        grad = alpha * (1 - K ** 2)[:, np.newaxis] * X
    elif kern['type'] == 'cosine':
        norm_xi = np.linalg.norm(x_i)
        norm_X = np.linalg.norm(X, axis=1)
        dot_product = X @ x_i
        denom = norm_xi * norm_X
        denom[denom == 0] = np.finfo(float).eps
        grad = (X / (norm_xi * norm_X[:, np.newaxis]) - \
                (dot_product / (norm_xi ** 3 * norm_X))[:, np.newaxis] * x_i)
    elif kern['type'] == 'linear':
        grad = X
    else:
        raise ValueError("Unsupported kernel type")
    return grad

def kern_compute(kern, X):
    n = X.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = kernel_function(X[i], X[j], kern)
    return A

def kern_grad_x(kern, x_i, X):
    return kernel_gradient(x_i, X, kern)

def plot_true_vs_predicted(X_true, X_predicted, kernel_name=''):
    """
    Plots a scatter plot of true vs. predicted values with a line showing perfect prediction.
    """
    n_features = X_true.shape[1]
    plt.figure(figsize=(8, 6))
    for i in range(n_features):
        plt.scatter(X_true[:, i], X_predicted[:, i], alpha=0.5, label=f'Feature {i+1}')
    
    max_val = max(np.max(X_true), np.max(X_predicted))
    min_val = min(np.min(X_true), np.min(X_predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs. Predicted Values ({kernel_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot_{kernel_name}.png')

# Load or generate your data
# Ensure you have 'aiida_ads_data_june21.csv' in your working directory.
df = pd.read_csv('aiida_ads_data_june21.csv')
df = df.select_dtypes(include=np.number)
data = df.to_numpy()
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std
X = data

# Choose kernel type and parameters

# For linear kernel only:
# kern = {'type': 'linear'}

# For a combination of linear and RBF kernels:
kern = {
    'type': 'combined',
    'kernels': [
        {'type': 'linear'},
        {'type': 'rbf', 'gamma': 5}
    ],
    'weights': [0.5, 0.5],  # Adjust weights as needed
    'combination': 'sum'    # 'sum' or 'product'
}

# Alternatively, select other kernels:
# kern = {'type': 'polynomial', 'degree': 3, 'coef0': 1, 'alpha': 1}
# kern = {'type': 'sigmoid', 'alpha': 0.5, 'coef0': 1}
# kern = {'type': 'cosine'}

# Create missing data
p = 0.40  # Proportion of missing data
seedling = 42
newX, I = kpca_misser(X, p, seedling)
# I = np.ones_like(newX)
# Set options
options = kpca_options()

# Initialize W and sigma
numComp = 25  # Should be less than the number of data points
A = kern_compute(kern, newX)
sigma, W, lambda_ = ppca(A, numComp)
print("Initialization complete")

# Perform KPCA reconstruction
updatedX, var, PcEig, PcCoeff = kpca_missing_data(newX, options, W, sigma, kern, I, numComp)

# Save PCs and coefficients
df_PCs = pd.DataFrame(PcEig)
df_PCs.to_csv('PCs.csv', index=False)

# Calculate the percentage of variance explained by the first 10 components
first_ten_sum = sum(PcCoeff[:10])
total_sum = sum(PcCoeff)
variance_explained = first_ten_sum / total_sum
print(f"Percentage of variance explained by first 10 components: {variance_explained*100:.2f}%")

# The updatedX is your reconstructed data
X_reconstructed = updatedX

# Compare original and reconstructed data
print("Original X shape:", X.shape)
print("Reconstructed X shape:", X_reconstructed.shape)

# Save original and reconstructed data to CSV files
X_df = pd.DataFrame(X)
X_reconstructed_df = pd.DataFrame(X_reconstructed)

kernel_name = kern['type']
if kernel_name == 'combined':
    kernel_name += '_' + '_'.join([k['type'] for k in kern['kernels']])

X_df.to_csv(f'X_{kernel_name}.csv', index=False)
X_reconstructed_df.to_csv(f'X_reconstructed_{kernel_name}.csv', index=False)

print(f"Files saved for {kernel_name} kernel as 'X_{kernel_name}.csv' and 'X_reconstructed_{kernel_name}.csv'.")

# Plot true vs. predicted values
plot_true_vs_predicted(X, X_reconstructed, kernel_name=kernel_name)
