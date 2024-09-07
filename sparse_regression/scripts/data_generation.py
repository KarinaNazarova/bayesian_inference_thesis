import numpy as np
import pandas as pd



def generate_synthetic_data(N=1000, num_active=6, num_inactive=4, betas_true=None, alpha_true=-1, distribution='normal', class_imbalance_ratio=None, correlation=0, seed=42):
    np.random.seed(seed)
    
    # Generate active features
    if distribution == 'normal':
        X_active = np.random.normal(0, 1, (N, num_active))

    # Calculate the logit and probabilities
    logit = np.dot(X_active, betas_true) + alpha_true
    probabilities = 1 / (1 + np.exp(-logit))
    
    # Get the binary outcome variable based on these probabilities
    y = np.random.binomial(1, probabilities)    

    # Introduce inactive features
    X_inactive = np.random.normal(0, 1, (N, num_inactive))

    # Combine active and inactive features
    X = np.hstack((X_active, X_inactive))

    # Combine features and outcome into the dataframe
    data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    data['outcome'] = y

    return data, betas_true, alpha_true



def generate_correlated_synthetic_data(N=1000, num_active=6, num_inactive=4, betas_true=None, alpha_true=-1, distribution='normal', class_imbalance_ratio=None, correlation=0, seed=42):
    np.random.seed(seed)
    
    # Define the mean vector for the features
    mean_vector = [0] * (num_active + num_inactive)

    # Define the covariance matrix (Identity)
    cov_matrix = np.eye(num_active + num_inactive)  

    # Introduce correlations among the active features
    if correlation > 0:
        for i in range(1, num_active):
            for j in range(i):
                cov_matrix[i, j] = cov_matrix[j, i] = correlation

    # Generate samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(mean_vector, cov_matrix, N)

    # Generate the logit and probabilities
    logit = np.dot(samples[:, :num_active], betas_true) + alpha_true
    probabilities = 1 / (1 + np.exp(-logit))

    # Sample the binary outcome variable based on these probabilities
    y = np.random.binomial(1, probabilities)    

    # Create DataFrame to hold the features and the outcome
    data = pd.DataFrame(samples, columns=[f'feature_{i+1}' for i in range(samples.shape[1])])
    data['outcome'] = y

    return data, betas_true, alpha_true
