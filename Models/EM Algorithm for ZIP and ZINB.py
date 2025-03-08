import numpy as np
from scipy.special import gammaln

def em_zip(data, K, max_iter=100, tol=1e-6):
    """
    EM algorithm for Zero-Inflated Poisson (ZIP) biclustering.
    """
    n, p = data.shape
    # Initialize parameters
    pi_k = np.random.dirichlet([1] * K)
    lambda_kj = np.random.rand(K, p)
    gamma_ik = np.zeros((n, K))
    
    for iteration in range(max_iter):
        # E-Step
        for i in range(n):
            for k in range(K):
                likelihood = np.prod([
                    pi_k[k] * (lambda_kj[k, j] ** data[i, j] * np.exp(-lambda_kj[k, j]) / np.math.factorial(data[i, j]))
                    for j in range(p)
                ])
                gamma_ik[i, k] = likelihood
            gamma_ik[i, :] /= np.sum(gamma_ik[i, :])
        
        # M-Step
        pi_k = np.mean(gamma_ik, axis=0)
        for k in range(K):
            for j in range(p):
                lambda_kj[k, j] = np.sum(gamma_ik[:, k] * data[:, j]) / np.sum(gamma_ik[:, k])
        
        # Check convergence
        if iteration > 0 and np.abs(pi_k - prev_pi_k).sum() < tol:
            break
        prev_pi_k = pi_k.copy()
    
    return pi_k, lambda_kj