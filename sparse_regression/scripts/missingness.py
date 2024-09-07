import numpy as np
import pandas as pd
from scipy.stats import mode

import stan 
import nest_asyncio
nest_asyncio.apply()

# reference: https://civil.colorado.edu/~balajir/CVEN6833/bayes-resources/RM-StatRethink-Bayes.pdf page 438

def impute_missing_values_and_save_df(data, filename):
    model_code = """
    data {
        int<lower=0> N;      // Number of observations
        int<lower=0> P;      // Number of predictors

        array[P] int num_missing; // Number of missing values for each predictor

        vector<lower=0, upper=1>[N] y;         // Response variable // must be either 0 or 1 
        matrix<lower=0>[N, P] X;       // Predictor matrix // each value must be bigger than 0
        array[N, P] int indicator_for_missing;    // Indicator for missing values in neocortex
    }
    parameters {
        real alpha;     // Intercept
        real<lower=0> sigma;        // Standard deviation of residuals
    
        vector[P] beta;
        
        matrix<lower=0>[max(num_missing), P] imputed_values;   // Imputed values for missing predictors, all values are greater than 0 
        vector[P] mu_predictors;      // Mean of the predictors
        vector<lower=0>[P] sigma_predictors;     // Standard deviation of predictors, all values are greater than 0
    }
    model {
        vector[N] mu;
        matrix[N, P] predictors_merged;

        alpha ~ normal(0,10);
        beta ~ normal(0,10);

        for (p in 1:P) {
            mu_predictors[p] ~ normal(0.5,1);
            sigma_predictors[p] ~ cauchy(0,1);
            
            // merge missing and observed
            for (i in 1:N){
                predictors_merged[i, p] = X[i, p];
                if (indicator_for_missing[i, p] > 0) predictors_merged[i, p] = imputed_values[indicator_for_missing[i, p], p];
            }

            // imputation
            predictors_merged[, p] ~ normal(mu_predictors[p], sigma_predictors[p]);
        }

        // regression
        for (i in 1:N) {
            mu[i] = alpha + dot_product(beta, predictors_merged[i]);
        }

        sigma ~ cauchy(0,1);
        y ~ normal(mu, sigma);
    }
    generated quantities {    
        matrix[N, P] predictors_merged_output;
        for (p in 1:P) {
            // merge missing and observed
            for (i in 1:N){
                predictors_merged_output[i, p] = X[i, p];
                if (indicator_for_missing[i, p] > 0) predictors_merged_output[i, p] = imputed_values[indicator_for_missing[i, p], p];
            }
        }
    }
    """

    # Prepare data for Stan
    y = data['cdi_label'].values
    X = data.drop(columns=['cdi_label']).values
    N = X.shape[0]
    P = X.shape[1]

    # Helper function to identify missing indices and count missing values
    def identify_missing_indices(data):
        missing = np.where(np.isnan(data), 1, 0)
        missing = np.array([missing[n] * np.sum(missing[:n+1]) for n in range(len(missing))])
        num_missing = np.count_nonzero(missing)
        return missing, num_missing

    # Identify missing indices and count missing values for each predictor
    indicator_for_missing = np.zeros((N, P), dtype=int)
    num_missing = np.zeros(P, dtype=int)


    for j in range(P):
        indicator_for_missing[:, j], num_missing[j] = identify_missing_indices(X[:, j])

    # Prepare data for Stan
    data_stan = {
        'N': N,
        'P': P,
        'num_missing': num_missing.tolist(),
        'y': y.tolist(),
        'X': np.nan_to_num(X).tolist(),
        'indicator_for_missing': indicator_for_missing.tolist()
    }

    # Build the model
    sm = stan.build(model_code, data=data_stan)

    # Sample from the model
    fit = sm.sample(num_chains=2, num_samples=2)

    # Extract imputed values
    predictors_merged_output = fit['predictors_merged_output']

    data_imputed = pd.DataFrame(predictors_merged_output[:, :, 0], columns=data.drop(columns=['cdi_label']).columns)
    data_imputed['cdi_label'] = y
    print(data_imputed.head())

    path = f'{filename}.xlsx'
    data_imputed.to_excel(path, index=False)
    print(f'Imputed data saved to {path}')



# ####### COMPARE STATISTICS #######
# # Calculate statistics for the imputed data
# stats_imputed = data_imputed.agg(['mean', 'std']).T
# stats_imputed.columns = ['mean_imputed', 'std_imputed']

# # Calculate statistics for the original data
# stats_original = data.agg(['mean', 'std']).T
# stats_original.columns = ['mean_original', 'std_original']

# # Combine the statistics into one DataFrame for comparison
# stats_comparison = pd.concat([stats_imputed, stats_original], axis=1)


# pd.set_option('display.max_rows', 100)
# # Print the comparison0
# print(stats_comparison)