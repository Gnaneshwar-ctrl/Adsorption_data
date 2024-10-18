import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def kpca(kern, X, dims):
    A = kern_compute(kern, X)
    sigma, V, lambda_ = ppca(A, dims)
    W = V
    return W

def kpca_like_missing(X_flat, W, sigma, kern, I, npts, Dim):
    X = X_flat.view(npts, Dim)
    A = kern_compute(kern, X)
    Kx = W @ W.t() + sigma * torch.eye(npts, device=device)
    sign, logdet_Kx = torch.slogdet(Kx)
    if sign <= 0:
        raise ValueError("Kx is not positive definite.")
    f = 0.5 * logdet_Kx + 0.5 * torch.trace(A @ torch.pinverse(Kx)) - npts / 2
    return f

def kpca_missing_data(X, options, W, sigma, kern, I, numComp):
    npts, Dim = X.shape
    diffKL = 1
    Iter = 0
    maxOuterIter = options['maxIter']
    sensitivity = options['sensitivity']
    
    X = X.clone().detach().requires_grad_(True).to(device)
    optimizer = optim.LBFGS([X], max_iter=10000, tolerance_grad=1e-5)
    
    while diffKL > sensitivity and Iter < maxOuterIter:
        Iter += 1
        OldKL = kpca_like_missing(X.flatten(), W, sigma, kern, I, npts, Dim)
        
        def closure():
            optimizer.zero_grad()
            f = kpca_like_missing(X.flatten(), W, sigma, kern, I, npts, Dim)
            f.backward()
            return f
        
        optimizer.step(closure)
        
        with torch.no_grad():
            NewKL = kpca_like_missing(X.flatten(), W, sigma, kern, I, npts, Dim)
            diffKL = OldKL.item() - NewKL.item()
        
        if diffKL < 0:
            raise ValueError('KL divergence increased!')
        
        A = kern_compute(kern, X)
        sigma, V, lambda_ = ppca(A, numComp)
        W = V @ torch.diag(torch.sqrt(lambda_))
        
        if options['display']:
            # Code to display objective function values (optional)
            pass
        
        print(f'Iteration {Iter} complete. KL divergence decreased by {diffKL}')
        
    if Iter == maxOuterIter:
        print('Maximum iterations exceeded.')
    
    updatedX = X.detach()
    var = sigma
    PcEig = V
    PcCoeff = lambda_
    return updatedX, var, PcEig, PcCoeff

def ppca(A, dims):
    lambda_, V = torch.linalg.eigh(A)
    idx = torch.argsort(lambda_, descending=True)
    lambda_ = lambda_[idx]
    V = V[:, idx]
    
    # Estimate sigma as the mean of the discarded eigenvalues
    if dims < len(lambda_):
        discarded_lambda = lambda_[dims:]
        sigma = torch.mean(discarded_lambda)
    else:
        sigma = torch.tensor(1e-5, device=device)
        print("Warning: dims >= number of eigenvalues. Setting sigma to a small positive value.")
    
    # Truncate to the desired number of dimensions
    lambda_ = lambda_[:dims]
    V = V[:, :dims]
    
    return sigma, V, lambda_

def kern_compute(kern, X):
    if kern['type'] == 'combined':
        combination = kern.get('combination', 'sum')
        kernels = kern['kernels']
        weights = kern.get('weights', [1]*len(kernels))
        if combination == 'sum':
            K_total = torch.zeros((X.shape[0], X.shape[0]), device=device)
            for k, w in zip(kernels, weights):
                K = kern_compute(k, X)
                K_total += w * K
            return K_total
        elif combination == 'product':
            K_total = torch.ones((X.shape[0], X.shape[0]), device=device)
            for k, w in zip(kernels, weights):
                K = kern_compute(k, X)
                K_total *= K ** w
            return K_total
        else:
            raise ValueError("Unsupported combination method")
    elif kern['type'] == 'rbf':
        gamma = kern.get('gamma', 1.0)
        X_norm = (X ** 2).sum(dim=1).view(-1, 1)
        K = torch.exp(-gamma * (X_norm + X_norm.t() - 2 * X @ X.t()))
        return K
    elif kern['type'] == 'linear':
        K = X @ X.t()
        return K
    else:
        raise ValueError("Unsupported kernel type")

def kpca_misser(X, p, seedling):
    if not 0 <= p <= 1:
        raise ValueError('p must be in [0,1]')
    torch.manual_seed(seedling)
    npts, Dim = X.shape
    Y = torch.rand(npts, Dim, device=device)
    I = (Y > (1 - p)).float()
    neuX = X * (1 - I)
    sum_I = torch.sum(1 - I, dim=0)
    sum_I[sum_I == 0] = 1  # Avoid division by zero
    avgX = torch.sum(neuX, dim=0) / sum_I
    Replace = torch.zeros((npts, Dim), device=device)
    for i in range(Dim):
        Replace[:, i] = avgX[i] * I[:, i]
    newX = neuX + Replace
    return newX, I

def kpca_options():
    options = {
        'maxIter': 5000,
        'sensitivity': 1e-4,
        'display': False,
        'captVar': 0.95,
    }
    return options

def plot_true_vs_predicted(X_true, X_predicted, kernel_name=''):
    """
    Plots a scatter plot of true vs. predicted values with a line showing perfect prediction.
    """
    X_true = X_true.cpu().numpy()
    X_predicted = X_predicted.cpu().numpy()
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
    plt.close()

# Load or generate your data
# Ensure you have 'aiida_ads_data_june21.csv' in your working directory.
df = pd.read_csv('aiida_ads_data_june21.csv')
df = df.select_dtypes(include='number')
data = df.to_numpy()
data = torch.tensor(data, dtype=torch.float32).to(device)
mean = torch.mean(data, dim=0)
std = torch.std(data, dim=0)
data = (data - mean) / std
X = data

# Choose kernel type and parameters
kern = {
    'type': 'combined',
    'kernels': [
        {'type': 'linear'},
        {'type': 'rbf', 'gamma': 5}
    ],
    'weights': [0.5, 0.5],  # Adjust weights as needed
    'combination': 'sum'    # 'sum' or 'product'
}

# Create missing data
p = 0.00  # Proportion of missing data
seedling = 42
newX, I = kpca_misser(X, p, seedling)

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
df_PCs = pd.DataFrame(PcEig.cpu().numpy())
df_PCs.to_csv('PCs.csv', index=False)

# Calculate the percentage of variance explained by the first 10 components
first_ten_sum = torch.sum(PcCoeff[:10])
total_sum = torch.sum(PcCoeff)
variance_explained = (first_ten_sum / total_sum).item()
print(f"Percentage of variance explained by first 10 components: {variance_explained*100:.2f}%")

# The updatedX is your reconstructed data
X_reconstructed = updatedX

# Compare original and reconstructed data
print("Original X shape:", X.shape)
print("Reconstructed X shape:", X_reconstructed.shape)

# Save original and reconstructed data to CSV files
X_df = pd.DataFrame(X.cpu().numpy())
X_reconstructed_df = pd.DataFrame(X_reconstructed.cpu().numpy())

kernel_name = kern['type']
if kernel_name == 'combined':
    kernel_name += '_' + '_'.join([k['type'] for k in kern['kernels']])

X_df.to_csv(f'X_{kernel_name}.csv', index=False)
X_reconstructed_df.to_csv(f'X_reconstructed_{kernel_name}.csv', index=False)

print(f"Files saved for {kernel_name} kernel as 'X_{kernel_name}.csv' and 'X_reconstructed_{kernel_name}.csv'.")

# Plot true vs. predicted values
plot_true_vs_predicted(X, X_reconstructed, kernel_name=kernel_name)
