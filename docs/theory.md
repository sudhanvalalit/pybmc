# Theoretical Background

This section provides a detailed explanation of the theoretical foundations of the `pybmc` package, including Bayesian Model Combination (BMC), Singular Value Decomposition (SVD) for orthogonalization, and the Gibbs sampling methods used for inference.

## Bayesian Model Combination (BMC)

Bayesian Model Combination is a statistical framework for combining predictions from multiple models. Instead of selecting a single "best" model, BMC computes a weighted average of all models, where the weights are determined by the models' performance on the observed data.

### Mathematical Formulation

Given a set of \(K\) models, \(M_1, M_2, \dots, M_K\), the combined prediction for a data point \(x\) is given by:

\[
y(x) = \sum_{k=1}^K w_k f_k(x)
\]

where:
- \(f_k(x)\) is the prediction of model \(M_k\) for input \(x\).
- \(w_k\) is the weight assigned to model \(M_k\), with the constraints that \(\sum_{k=1}^K w_k = 1\) and \(w_k \ge 0\).

In the Bayesian framework, we treat the weights \(\mathbf{w} = (w_1, \dots, w_K)\) as random variables and aim to infer their posterior distribution given the observed data \(D\). Using Bayes' theorem, the posterior distribution is:

\[
p(\mathbf{w} | D) \propto p(D | \mathbf{w}) p(\mathbf{w})
\]

where \(p(D | \mathbf{w})\) is the likelihood of the data given the weights, and \(p(\mathbf{w})\) is the prior distribution of the weights.

## Orthogonalization with Singular Value Decomposition (SVD)

In practice, the predictions from different models are often highly correlated. This collinearity can lead to unstable estimates of the model weights and can cause overfitting. To address this, `pybmc` uses Singular Value Decomposition (SVD) to orthogonalize the model predictions before performing Bayesian inference.

SVD decomposes the matrix of centered model predictions \(X\) into three matrices:

\[
X = U S V^T
\]

where:
- \(U\) is an \(N \times N\) orthogonal matrix whose columns are the left singular vectors.
- \(S\) is an \(N \times K\) rectangular diagonal matrix with the singular values on the diagonal.
- \(V^T\) is a \(K \times K\) orthogonal matrix whose rows are the right singular vectors.

By keeping only the first \(m \ll K\) singular values and vectors, we can create a low-rank approximation of \(X\) that captures the most important variations in the model predictions while filtering out noise and redundancy. This results in a more stable and robust inference process.

## Gibbs Sampling for Inference

`pybmc` uses Gibbs sampling to draw samples from the posterior distribution of the model weights and other parameters. Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm that iteratively samples from the conditional distribution of each parameter given the current values of all other parameters.

### Standard Gibbs Sampler

The standard Gibbs sampler in `pybmc` assumes a Gaussian likelihood and conjugate priors for the model parameters. The algorithm iteratively samples from the full conditional distributions of the regression coefficients (related to the model weights) and the error variance.

### Gibbs Sampler with Simplex Constraints

`pybmc` also provides a Gibbs sampler that enforces simplex constraints on the model weights (i.e., \(\sum w_k = 1\) and \(w_k \ge 0\)). This is achieved by performing a random walk in the space of the transformed parameters and using a Metropolis-Hastings step to accept or reject proposals that fall outside the valid simplex region.