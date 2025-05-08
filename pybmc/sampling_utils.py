import numpy as np

def coverage(self, percentiles, rndm_m, models_output):
    """
    Calculate the coverage of credible intervals for the model's predictions.

    :param percentiles: List of percentiles to evaluate the credible intervals.
    :param rndm_m: Array of random samples of model predictions.
    :param models_output: DataFrame containing the true values under the "truth" column.
    :return: List of coverage percentages for each percentile.
    """
    #  How often the model’s credible intervals actually contain the true value
    data_total = len(rndm_m.T)   # Number of data points
    M_evals = len(rndm_m)        # Number of samples
    data_true = models_output["truth"].tolist()

    coverage_results = []

    for p in percentiles:
        count_covered = 0
        for i in range(data_total):
            # Sort model evaluations for the i-th data point
            sorted_evals = np.sort(rndm_m.T[i])
            # Find indices for lower and upper bounds of the credible interval
            lower_idx = int((0.5 - p / 200) * M_evals)
            upper_idx = int((0.5 + p / 200) * M_evals) - 1
            # Check if the true value y[i] is within this interval
            if sorted_evals[lower_idx] <= data_true[i] <= sorted_evals[upper_idx]:
                count_covered += 1
        coverage_results.append(count_covered / data_total * 100)

    return coverage_results

def rndm_m_random_calculator(self, filtered_model_predictions, samples):
    """
    Calculate random samples of model predictions and their credible intervals.

    :param filtered_model_predictions: Matrix of filtered model predictions for isotopes.
    :param samples: Array of sampled parameter sets from the Gibbs sampler.
    :return: A tuple containing:
             - rndm_m: Array of random samples of model predictions.
             - [lower_radius, median_radius, upper_radius]: Percentile-based credible intervals.
    """
    np.random.seed(142857)
    rng = np.random.default_rng()
    
    theta_rand_selected = rng.choice(samples, 10000, replace = False)

    model_weights_random = []
    for beta in theta_rand_selected:
        model_weights_random.append(np.dot(beta[:-1], self.Vt_hat) + np.full(len(self.Vt_hat[0]) , 1/len(self.Vt_hat[0])))
    model_weights_random = np.array(model_weights_random)

    rndm_m = []
    for i in range(len(model_weights_random)):
        yvals_rand_radius= filtered_model_predictions.dot(model_weights_random[i].T)

    
        rndm_m.append(yvals_rand_radius +
                    np.random.multivariate_normal(np.full(
                        len(yvals_rand_radius)
                        ,0), np.diag(1.0 * np.full(len(yvals_rand_radius),1.0 * theta_rand_selected[i][-1]**2 ) ))) 
    rndm_m = np.array(rndm_m)

        
    lower_radius = np.percentile(rndm_m, 2.5, axis = 0)
    median_radius = np.percentile(rndm_m, 50, axis = 0)
    upper_radius = np.percentile(rndm_m, 97.5, axis = 0)

    return rndm_m, [lower_radius, median_radius, upper_radius]