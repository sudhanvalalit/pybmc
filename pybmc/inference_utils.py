import numpy as np
import sys

def gibbs_sampler(y, X, iterations, prior_info):
    """
    Perform Gibbs sampling for Bayesian linear regression.

    :param y: Response vector.
    :param X: Design matrix.
    :param iterations: Number of iterations for the Gibbs sampler.
    :param prior_info: Tuple containing prior information (b_mean_prior, b_mean_cov, nu0, sigma20).
    :return: Array of sampled parameter sets.
    """
    
    b_mean_prior, b_mean_cov, nu0, sigma20 = prior_info
    b_mean_cov_inv = np.linalg.inv(b_mean_cov)
    n = len(y)
    # Precompute matrices
    X_T_X = X.T.dot(X)
    X_T_X_inv = np.linalg.inv(X_T_X)
    b_data = X_T_X_inv.dot(X.T).dot(y)

    # Initialize residuals and variance
    supermodel = X.dot(b_data)
    residuals = y - supermodel
    sigma2 = np.sum(residuals**2) / len(residuals)

    samples = []

def USVt_hat_extraction(U, S, Vt, components_kept):
    """
    Extract reduced-dimensionality matrices from Singular Value Decomposition (SVD).

    :param U: Left singular vectors from SVD.
    :param S: Singular values from SVD.
    :param Vt: Right singular vectors (transposed) from SVD.
    :param components_kept: Number of principal components to retain.
    :return: A tuple containing:
             - U_hat: Reduced left singular vectors.
             - S_hat: Retained singular values.
             - Vt_hat: Normalized right singular vectors.
             - Vt_hat_normalized: Original right singular vectors without normalization.
    """
    U_hat = np.array([U.T[i] for i in range(components_kept)]).T
    S_hat = S[:components_kept]
    Vt_hat = np.array([Vt[i] / S[i] for i in range(components_kept)])
    Vt_hat_normalized = np.array([Vt[i] for i in range(components_kept)])
    return U_hat, S_hat, Vt_hat, Vt_hat_normalized