import numpy as np
import pandas as pd
import os
import arviz as az

from sparse_regression.scripts.data_generation import generate_synthetic_data
from sparse_regression.models.logistic_model_fitting import fit_model, SparseModelType
from sparse_regression.scripts.plot_posterior import plot_posterior_distribution
from sparse_regression.scripts.evaluate import check_credible_intervals, calculate_confusion_matrix_metrics, plot_and_save_roc_curve, identify_active_and_inactive_features, compare_loo_scores_and_save_the_png, plot_total_and_correctly_identified_active_inactive_features

def run_experiment_A():
    experiment_name = 'A'

    ##### SPECIFY SCENARIOS TO TEST AND CREATE THE CSV FILES #####
    # Scenario Set 1: Normal distribution, no class imbalance, no correlation, different sample sizes
    scenario_set_1 = [
        {"N": 10, "num_active": 6, "num_inactive": 4, "betas_true": [-0.09, 0.2, -0.8, 1, -4, 7], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},
        {"N": 100, "num_active": 6, "num_inactive": 4, "betas_true": [-0.09, 0.2, -0.8, 1, -4, 7], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},
        {"N": 1000, "num_active": 6, "num_inactive": 4, "betas_true": [-0.09, 0.2, -0.8, 1, -4, 7], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},
        {"N": 10000, "num_active": 6, "num_inactive": 4, "betas_true": [-0.09, 0.2, -0.8, 1, -4, 7], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0}]

    # Scenario Set 2: Normal distribution, class imbalance, no correlation, different proportion of active and inactive features
    scenario_set_2 = [
        {"N": 1000, "num_active": 2, "num_inactive": 8, "betas_true": [-0.01, 0.8], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # lots inactive and some active
        {"N": 1000, "num_active": 8, "num_inactive": 2, "betas_true": [-0.01, 0.09, -0.5, 0.8, -1, 2, -3, 6], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # lots active and some inactive
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.01, 0.09, -0.5, 0.8, 6], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0}]

    # Scenario Set 3: Normal distribution, no class imbalance, no correlation, different effect sizes
    scenario_set_3 = [
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.01, 0.02, 0.03, -0.04, 0.08], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all small
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.2, 0.3, -0.4, 0.5, 0.9], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all medium
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-1.5, -3, 5, 7, 8], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all large
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.01, 0.05, 0.3, 0.8, 5], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0}  # mixed effects
    ]

    scenario_set_3 = [
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.01, 0.02, 0.03, -0.04, 0.08], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all small
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.2, 0.3, -0.4, 0.5, 0.9], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all medium
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-1.5, -3, 5, 7, 8], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0},  # all large
        {"N": 1000, "num_active": 5, "num_inactive": 5, "betas_true": [-0.09, 0.3, 0.5, 2, 7], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0}  # mixed effects
    ]

    # scenario_set_4 = [{"N": 1000, "num_active": 6, "num_inactive": 4, "betas_true": [0.3, 0.4, 1.0, 2.0, 10.0, 12.0], "alpha_true": -1, "distribution": "normal", "class_imbalance_ratio": None, "correlation": 0}]

    scenarios = scenario_set_1 + scenario_set_2 + scenario_set_3 
    
    models_to_test = [SparseModelType.NORMAL, SparseModelType.LAPLACE, SparseModelType.CAUCHY, SparseModelType.HORSESHOE, SparseModelType.HORSESHOE_REGULARISED]

    results = []
   
    # Generate synthetic data for each scenario and save to CSV
    for i, scenario in enumerate(scenarios):
        loo_scores = []
        for model_type in models_to_test:
            scenario_name = f"scenario_{i+1}"
            model_name = model_type.name.lower()
            print(f"Running scenario {scenario_name} with model {model_name}")

            # Generate synthetic data
            data, betas_true, alpha_true = generate_synthetic_data(**scenario)
            
            filename = f'sparse_regression/data/experiment_{experiment_name}/synthetic_data_scenario_{i+1}.csv'
            os.makedirs(f'sparse_regression/data/experiment_{experiment_name}/', exist_ok=True)
        
            data.to_csv(filename, index=False)
            print(f"Synthetic data generated and saved to '{filename}'.")

            fit = fit_model(data, model_type)
            
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

            # Plot the posterior distributions
            plot_posterior_distribution(alpha_samples, beta_samples, alpha_true, betas_true, experiment_name, scenario_name, model_name)

            # checks how many active/inactive features are satisfied based on the 95% credible intervals
            ci_results = check_credible_intervals(alpha_samples, beta_samples, alpha_true, betas_true, scenario["num_active"])

            # Calculate confusion matrixv and accuracy/f1 metrics
            X = data.drop('outcome', axis=1).values
            y = data['outcome'].values
            confusion_metrics = calculate_confusion_matrix_metrics(X, y, alpha_samples, beta_samples)

            plot_and_save_roc_curve(X, y, alpha_samples, beta_samples, experiment_name, model_name, experiment_name, scenario_name, dataset_name=None)

            active_features, inactive_features = identify_active_and_inactive_features(data, beta_samples, experiment_name, model_name, experiment_name, scenario_name, dataset_name=None)

            results.append({
                "scenario": scenario_name,
                "model": model_name,
                **ci_results,
                **confusion_metrics
            })
        loo_scores_df = pd.DataFrame(loo_scores).sort_values(by='elpd_loo', ascending=False)
        loo_scores_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/plots/{scenario_name}/loo_scores.csv', index=False)

        compare_loo_scores_and_save_the_png(loo_scores_df, experiment_name, scenario_name=scenario_name)

    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    os.makedirs(f'sparse_regression/results/experiment_{experiment_name}/', exist_ok=True)
        
    results_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/evaluation_metrics.csv', index=False)
    
    # save the graph for correctly identified active/inactive features
    plot_total_and_correctly_identified_active_inactive_features(results_df, experiment_name)
    print(f'Finished running the experiment {experiment_name} on scenario {scenario}.')