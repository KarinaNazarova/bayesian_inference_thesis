import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import os 
import arviz as az

from sparse_regression.models.logistic_model_fitting import fit_model, SparseModelType
from sparse_regression.scripts.plot_posterior import plot_posterior_distribution
from sparse_regression.scripts.evaluate import check_credible_intervals, calculate_confusion_matrix_metrics, plot_and_save_roc_curve, identify_active_and_inactive_features, compare_loo_scores_and_save_the_png

def run_experiment_B():
    experiment_name = 'B'

    ##### CREATE SYNTHETIC DATASET AND PREPARE IT FOR MODEL TESTING #####
    # Simulated data that resembles real data 
    data = pd.read_excel('sparse_regression/data/processed/synthetic_data_copula.xlsx', index_col=None, header=0)
    
    # Select important and not important features based on previous results
    columns_important_features = ['panas_negative', 'tdearitm', 'ls_self', 'ls_family', 'eev3_product', 'eev29_product']
    columns_not_important_features = ['ls_comparative_self', 'eev9_product', 'eev52_product', 'eev54_product']

    columns_to_select = columns_important_features + columns_not_important_features
    data_with_selected_features = data[columns_to_select]

    # Standardise the features
    columns_to_scale = [col for col in columns_to_select if col not in ['eev3_product', 'eev29_product', 'eev9_product', 'eev52_product', 'eev54_product']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_with_selected_features[columns_to_scale])

    # Convert scaled features back to DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=columns_to_scale)

    # Combine with non-scaled (categorical) features
    for col in columns_to_select:
        if col not in columns_to_scale:
            scaled_features_df[col] = data_with_selected_features[col].values

    # Reorder columns to match original order
    data_with_selected_features = scaled_features_df[columns_to_select]


    # Define coefficients
    betas_true = [-0.09, 0.1, -0.8, -1, 1.5, 2]
    alpha_true = -1

    # Calculate logits and probabilities
    X_active = data_with_selected_features[columns_important_features].values
    logit = np.dot(X_active, betas_true) + alpha_true
    probabilities = 1 / (1 + np.exp(-logit))

    # Generate binary outcomes
    y = np.random.binomial(1, probabilities)

    # Combine data and outcome into one DataFrame
    data_with_selected_features['outcome'] = y

    print("Data with selected features and outcome label distribution:")
    print(data_with_selected_features['outcome'].value_counts())

    filename = f'sparse_regression/data/experiment_{experiment_name}/synthetic_data.csv'
    os.makedirs(f'sparse_regression/data/experiment_{experiment_name}/', exist_ok=True)
    data_with_selected_features.to_csv(filename, index=False)
    print(f"Synthetic data generated and saved to '{filename}'.")

    ##### END CREATE SYNTHETIC DATASET AND PREPARE IT FOR MODEL TESTING #####


    ##### TEST THE SPARSE MODELS ON THE CREATED DATASET #####
    models_to_test = [SparseModelType.NORMAL, SparseModelType.LAPLACE, SparseModelType.CAUCHY, SparseModelType.HORSESHOE, SparseModelType.HORSESHOE_REGULARISED]

    results = []
    loo_scores = []
    for model_type in models_to_test:
        model_name = model_type.name.lower()
        print(f"Running model {model_name}")

        fit = fit_model(data_with_selected_features, model_type)

        # Calculate LOO score
        idata = az.from_pystan(posterior=fit,
                               log_likelihood={"y": "log_lik"})
        loo_result = az.loo(idata, pointwise=True)
        waic_result = az.waic(idata, pointwise=True)
            
        loo_scores.append({
                    "Model": model_name,
                    "elpd_loo": loo_result.elpd_loo,
                    "se_elpd_loo": loo_result.se,
                    "p_loo": loo_result.p_loo,
                    'elpd_waic': waic_result.elpd_waic,
                    'se_elpd_waic': waic_result.se,
                    'p_waic': waic_result.p_waic,
                    'idata': idata
                })

        # Extract posterior samples
        beta_samples = fit['beta']
        alpha_samples = fit['alpha']

        # Calculate mean for alpha and beta
        alpha_mean = np.mean(alpha_samples)
        beta_means = np.mean(beta_samples.T, axis=0)

        # Plot posterior distributions
        plot_posterior_distribution(alpha_samples, beta_samples, alpha_true, betas_true, experiment_name, 'synthetic_data', model_name)

        ci_results = check_credible_intervals(alpha_samples, beta_samples, alpha_true, betas_true, len(columns_important_features))

        # Calculate confusion matrix and accuracy/f1 metrics
        X = data_with_selected_features.drop('outcome', axis=1).values
        y = data_with_selected_features['outcome'].values
        confusion_metrics = calculate_confusion_matrix_metrics(X, y, alpha_samples, beta_samples)

        plot_and_save_roc_curve(X, y, alpha_samples, beta_samples, experiment_name, model_name, experiment_name, 'synthetic_data', dataset_name=None)

        # Identify active and inactive features (active if 95% CI for beta does not contain zero)
        active_features, inactive_features = identify_active_and_inactive_features(data_with_selected_features, beta_samples, experiment_name, model_name, experiment_name, 'synthetic_data', dataset_name=None)

        results.append({
            "model": model_name,
            **ci_results,
            **confusion_metrics
        })
    os.makedirs(f'sparse_regression/results/experiment_{experiment_name}', exist_ok=True)

    loo_scores_df = pd.DataFrame(loo_scores).sort_values(by='elpd_loo', ascending=False)
    loo_scores_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/loo_scores.csv', index=False)
    compare_loo_scores_and_save_the_png(loo_scores_df, experiment_name, dataset_name=None, scenario_name=None)

    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.colheader_justify', 'left')
    pd.set_option('display.width', None)

    results_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/evaluation_metrics.csv', index=False)
    print(f'Finished running the experiment {experiment_name}.')