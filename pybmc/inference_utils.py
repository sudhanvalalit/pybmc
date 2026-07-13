import numpy as np
from scipy import stats

from .error_models import VARIANCE_FLOOR


def gibbs_sampler(y, X, iterations, prior_info):
    """
    Performs Gibbs sampling for Bayesian linear regression.

    Args:
        y (numpy.ndarray): Response vector (centered).
        X (numpy.ndarray): Design matrix.
        iterations (int): Number of sampling iterations.
        prior_info (tuple[numpy.ndarray, numpy.ndarray, float, float]): Prior parameters:
            - `b_mean_prior` (numpy.ndarray): Prior mean for coefficients.
            - `b_mean_cov` (numpy.ndarray): Prior covariance matrix.
            - `nu0` (float): Prior degrees of freedom for variance.
            - `sigma20` (float): Prior variance.

    Returns:
        numpy.ndarray: Posterior samples `[beta, sigma]`.
    """
    b_mean_prior, b_mean_cov, nu0, sigma20 = prior_info
    b_mean_cov_inv = np.linalg.inv(b_mean_cov)
    n = len(y)

    X_T_X = X.T.dot(X)
    X_T_X_inv = np.linalg.inv(X_T_X)

    b_data = X_T_X_inv.dot(X.T).dot(y)
    supermodel = X.dot(b_data)
    residuals = y - supermodel
    sigma2 = np.sum(residuals**2) / len(residuals)
    cov_matrix = sigma2 * X_T_X_inv

    samples = []

    # Initialize sigma2 with a small positive value to avoid division by zero
    sigma2 = max(sigma2, 1e-6)

    for i in range(iterations):
        # Regularize the covariance matrix to ensure it is positive definite
        cov_matrix = np.linalg.inv(X_T_X / sigma2 + b_mean_cov_inv + np.eye(X_T_X.shape[0]) * 1e-6)
        mean_vector = cov_matrix.dot(
            b_mean_cov_inv.dot(b_mean_prior) + X.T.dot(y) / sigma2
        )
        b_current = np.random.multivariate_normal(mean_vector, cov_matrix)

        # Sample from the conditional posterior of sigma2 given bs and data
        supermodel = X.dot(b_current)
        residuals = y - supermodel
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 + np.sum(residuals**2)) / 2.0
        sigma2 = max(1 / np.random.gamma(shape_post, 1 / scale_post), 1e-6)

        samples.append(np.append(b_current, np.sqrt(sigma2)))

    return np.array(samples)


def gibbs_sampler_simplex(
    y, X, Vt_hat, S_hat, iterations, prior_info, burn=10000, stepsize=0.001
):
    """
    Performs Gibbs sampling with simplex constraints on model weights.

    Args:
        y (numpy.ndarray): Centered response vector.
        X (numpy.ndarray): Design matrix of principal components.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.
        S_hat (numpy.ndarray): Singular values.
        iterations (int): Number of sampling iterations.
        prior_info (list[float]): `[nu0, sigma20]` - prior parameters for variance.
        burn (int, optional): Burn-in iterations (default: 10000).
        stepsize (float, optional): Proposal step size (default: 0.001).

    Returns:
        numpy.ndarray: Posterior samples `[beta, sigma]`.
    """
    bias0 = np.full(len(Vt_hat.T), 1 / len(Vt_hat.T))
    nu0, sigma20 = prior_info
    cov_matrix_step = np.diag(S_hat**2 * stepsize**2)
    n = len(y)
    b_current = np.full(len(X.T), 0)
    supermodel_current = X.dot(b_current)
    residuals_current = y - supermodel_current
    log_likelihood_current = -np.sum(residuals_current**2)
    sigma2 = -log_likelihood_current / len(residuals_current)
    samples = []
    acceptance = 0

    # Validate inputs
    if burn < 0:
        raise ValueError("Burn-in iterations must be non-negative.")
    if stepsize <= 0:
        raise ValueError("Stepsize must be positive.")

    # Burn-in phase
    for i in range(burn):
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        omegas_proposed = np.dot(b_proposed, Vt_hat) + bias0

        # Skip proposals with negative weights
        if not np.any(omegas_proposed < 0):
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed
            log_likelihood_proposed = -np.sum(residuals_proposed**2)
            acceptance_prob = min(
                1,
                np.exp((log_likelihood_proposed - log_likelihood_current) / sigma2),
            )
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current = log_likelihood_proposed

        # Sample variance
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 - log_likelihood_current) / 2.0
        sigma2 = 1 / np.random.gamma(shape_post, 1 / scale_post)

    # Sampling phase
    for i in range(iterations):
        b_proposed = np.random.multivariate_normal(b_current, cov_matrix_step)
        omegas_proposed = np.dot(b_proposed, Vt_hat) + bias0

        if not np.any(omegas_proposed < 0):
            supermodel_proposed = X.dot(b_proposed)
            residuals_proposed = y - supermodel_proposed
            log_likelihood_proposed = -np.sum(residuals_proposed**2)
            acceptance_prob = min(
                1,
                np.exp((log_likelihood_proposed - log_likelihood_current) / sigma2),
            )
            if np.random.uniform() < acceptance_prob:
                b_current = np.copy(b_proposed)
                log_likelihood_current = log_likelihood_proposed
                acceptance += 1

        # Sample variance
        shape_post = (nu0 + n) / 2.0
        scale_post = (nu0 * sigma20 - log_likelihood_current) / 2.0
        sigma2 = 1 / np.random.gamma(shape_post, 1 / scale_post)
        samples.append(np.append(b_current, np.sqrt(sigma2)))

    return np.array(samples)


def gibbs_sampler_heteroscedastic(
    y,
    X,
    variance_basis,
    iterations,
    burn=5000,
    proposal_scales=None,
    init_params=None,
    prior_spec=None,
    b_mean_prior=None,
    b_mean_cov=None,
    adapt_proposal=True,
    target_acceptance=0.25,
):
    """
    Gibbs-within-Metropolis sampler for heteroscedastic error models.

    The regression model is ``y_i ~ N(X_i . b, sigma_i^2)`` with a
    per-point variance that is linear in basis functions of the
    heteroscedasticity metrics:

        ``sigma_i^2 = variance_basis[i] . theta``

    The coefficients ``b`` are updated with a conjugate (weighted
    least-squares) Gibbs step; the variance parameters ``theta`` with a
    positivity-constrained Gaussian random-walk Metropolis-Hastings step
    under Gamma priors.

    Args:
        y (numpy.ndarray): Centered response vector, shape ``(n,)``.
        X (numpy.ndarray): Design matrix (principal components), shape
            ``(n, k)``.
        variance_basis (numpy.ndarray): Variance design matrix ``phi``
            with a leading column of ones, shape ``(n, p)``. See
            :func:`pybmc.error_models.variance_basis`.
        iterations (int): Number of retained posterior samples.
        burn (int, optional): Burn-in iterations discarded before
            retention (default: 5000).
        proposal_scales (list[float], optional): Diagonal of the Gaussian
            random-walk proposal covariance for ``theta`` (length p).
            Defaults to ``[1e-2, 1e-3, ..., 1e-3]``.
        init_params (list[float], optional): Initial values for the
            non-constant entries of ``theta`` (length p - 1). The
            constant term starts at the OLS residual variance. Defaults
            to 0.01 for every term.
        prior_spec (list[tuple[float, float]], optional): Gamma prior
            ``(shape, scale)`` for each entry of ``theta`` (length p).
            Defaults to ``(2, 10)`` for every parameter.
        b_mean_prior (numpy.ndarray, optional): Prior mean for ``b``
            (default zeros).
        b_mean_cov (numpy.ndarray, optional): Prior covariance for ``b``
            (default ``1e6 * I``, i.e. weakly informative).
        adapt_proposal (bool, optional): If True (default), rescale the
            proposal covariance during burn-in toward
            ``target_acceptance``, so the fixed defaults work across
            data scales. Adaptation stops at the end of burn-in, which
            preserves detailed balance for the retained samples.
        target_acceptance (float, optional): Acceptance rate targeted by
            the burn-in adaptation (default: 0.25).

    Returns:
        tuple[numpy.ndarray, float]:
            - `samples` (numpy.ndarray): Posterior samples
              ``[b_1..b_k, theta_1..theta_p]``, shape
              ``(iterations, k + p)``.
            - `acceptance_rate` (float): Post-burn-in MH acceptance rate.
    """
    if burn < 0:
        raise ValueError("Burn-in iterations must be non-negative.")

    n_points, n_betas = X.shape
    n_params = variance_basis.shape[1]

    if not np.allclose(variance_basis[:, 0], 1.0):
        raise ValueError(
            "The first column of 'variance_basis' must be ones "
            "(the constant variance term)."
        )

    if proposal_scales is None:
        proposal_scales = [1e-2] + [1e-3] * (n_params - 1)
    if len(proposal_scales) != n_params:
        raise ValueError(
            f"'proposal_scales' must have length {n_params} "
            f"(got {len(proposal_scales)})."
        )
    proposal_cov = np.diag(np.asarray(proposal_scales, dtype=float))

    if prior_spec is None:
        prior_spec = [(2, 10)] * n_params
    if len(prior_spec) != n_params:
        raise ValueError(
            f"'prior_spec' must have length {n_params} "
            f"(got {len(prior_spec)})."
        )
    priors = [stats.gamma(a=a, scale=scale) for a, scale in prior_spec]

    if b_mean_prior is None:
        b_mean_prior = np.zeros(n_betas)
    if b_mean_cov is None:
        b_mean_cov = np.eye(n_betas) * 1e6
    b_mean_cov_inv = np.linalg.inv(b_mean_cov)

    b_current = np.linalg.lstsq(X, y, rcond=None)[0]
    if init_params is None:
        init_params = [0.01] * (n_params - 1)
    if len(init_params) != n_params - 1:
        raise ValueError(
            f"'init_params' must have length {n_params - 1} "
            f"(got {len(init_params)})."
        )
    theta_current = np.concatenate(
        [[max(np.var(y - X @ b_current), VARIANCE_FLOOR)], init_params]
    )

    samples = []
    accept_count = 0

    # Burn-in adaptation of the proposal scale (Robbins-Monro style):
    # every `adapt_interval` iterations the covariance is rescaled toward
    # the target acceptance rate, then frozen for the sampling phase.
    proposal_scale_factor = 1.0
    adapt_interval = 100
    accept_recent = 0

    for i in range(burn + iterations):
        # --- Gibbs step: b | theta (weighted least squares) ---
        sigma2 = variance_basis @ theta_current
        sigma2[sigma2 <= VARIANCE_FLOOR] = VARIANCE_FLOOR
        Xw = X / sigma2[:, None]
        b_post_cov = np.linalg.inv(
            X.T @ Xw + b_mean_cov_inv + np.eye(n_betas) * 1e-6
        )
        b_post_mean = b_post_cov @ (
            Xw.T @ y + b_mean_cov_inv @ b_mean_prior
        )
        b_current = np.random.multivariate_normal(b_post_mean, b_post_cov)

        # --- MH step: theta | b (positivity-constrained random walk) ---
        residuals = y - X @ b_current
        theta_proposed = np.atleast_1d(
            np.random.multivariate_normal(
                theta_current, proposal_cov * proposal_scale_factor**2
            )
        )
        if np.all(theta_proposed > 0):
            scale_current = np.sqrt(
                np.maximum(variance_basis @ theta_current, VARIANCE_FLOOR)
            )
            scale_proposed = np.sqrt(
                np.maximum(variance_basis @ theta_proposed, VARIANCE_FLOOR)
            )
            log_lik_current = np.sum(
                stats.norm.logpdf(residuals, scale=scale_current)
            )
            log_lik_proposed = np.sum(
                stats.norm.logpdf(residuals, scale=scale_proposed)
            )
            log_prior_current = sum(
                p.logpdf(v) for p, v in zip(priors, theta_current)
            )
            log_prior_proposed = sum(
                p.logpdf(v) for p, v in zip(priors, theta_proposed)
            )
            log_ratio = (log_lik_proposed + log_prior_proposed) - (
                log_lik_current + log_prior_current
            )
            if np.log(np.random.uniform()) < log_ratio:
                theta_current = theta_proposed
                if i >= burn:
                    accept_count += 1
                else:
                    accept_recent += 1

        if adapt_proposal and i < burn and (i + 1) % adapt_interval == 0:
            recent_rate = accept_recent / adapt_interval
            proposal_scale_factor *= np.exp(recent_rate - target_acceptance)
            proposal_scale_factor = float(
                np.clip(proposal_scale_factor, 1e-6, 1e6)
            )
            accept_recent = 0

        if i >= burn:
            samples.append(np.concatenate([b_current, theta_current]))

    acceptance_rate = accept_count / max(iterations, 1)
    return np.array(samples), acceptance_rate


def USVt_hat_extraction(U, S, Vt, components_kept):
    """
    Extracts reduced-dimensionality matrices from SVD results.

    Args:
        U (numpy.ndarray): Left singular vectors.
        S (numpy.ndarray): Singular values.
        Vt (numpy.ndarray): Right singular vectors (transposed).
        components_kept (int): Number of components to retain.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - `U_hat` (numpy.ndarray): Reduced left singular vectors.
            - `S_hat` (numpy.ndarray): Retained singular values.
            - `Vt_hat` (numpy.ndarray): Normalized right singular vectors.
            - `Vt_hat_normalized` (numpy.ndarray): Original right singular vectors.
    """
    U_hat = np.array([U.T[i] for i in range(components_kept)]).T
    S_hat = S[:components_kept]
    Vt_hat = np.array([Vt[i] / S[i] for i in range(components_kept)])
    Vt_hat_normalized = np.array([Vt[i] for i in range(components_kept)])
    return U_hat, S_hat, Vt_hat, Vt_hat_normalized
