# Theoretical Background

This section will provide a short explanation of the Bayesian Model Combination (BMC) methodology used in pyBMC.

## Bayesian Model Combination

Bayesian Model Combination (BMC) is a method for combining multiple predictive models in a Bayesian framework. The key idea is to treat the model combination weights as random variables and infer their posterior distribution given the data.

### Mathematical Formulation

Given a set of models \( M_1, M_2, \dots, M_K \), the combined prediction for a data point \( x \) is:

\[
y = \sum_{k=1}^K w_k f_k(x)
\]

where:
- \( f_k(x) \) is the prediction of model \( M_k \) for input \( x \)
- \( w_k \) is the weight assigned to model \( M_k \), with \( \sum_{k=1}^K w_k = 1 \) and \( w_k \geq 0 \)

### Bayesian Inference

We place a prior distribution on the weights \( w \) and update this prior using observed data to obtain the posterior distribution:

\[
p(w | D) \propto p(D | w) p(w)
\]

where \( D \) is the observed data.

### Gibbs Sampling

We use Gibbs sampling to approximate the posterior distribution of the weights. The Gibbs sampler iteratively samples each weight conditional on the current values of the other weights and the data.

## Orthogonalization

To address collinearity between model predictions, we perform an orthogonalization step using Singular Value Decomposition (SVD). This transforms the model predictions into a set of orthogonal basis vectors, which improves the stability of the Bayesian inference and prevents overfitting.

