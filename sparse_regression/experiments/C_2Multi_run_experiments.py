import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import os 
import arviz as az

from sparse_regression.models.logistic_model_fitting import fit_model, SparseModelType
from sparse_regression.scripts.plot_posterior import  plot_posterior_distribution_real
from sparse_regression.scripts.evaluate import  calculate_confusion_matrix_metrics, identify_active_and_inactive_features, plot_and_save_roc_curve, identify_active_and_inactive_features, compare_loo_scores_and_save_the_png

def run_experiment_C():
    experiment_name = 'C2'
    dataset_names = ['nocdi_standardise_encode_recode_rounded_data_T1', 'standardise_encode_recode_rounded_data_T1']

    for dataset_name in dataset_names:
        ##### TEST THE MODELS ON A REAL DATA #####
        # Load real data without grouping questionnaires by test (with imputed missing values)
        # data = pd.read_excel(f'../{dataset_name}.xlsx', index_col=None, header=0)
        data = pd.read_excel(f'sparse_regression/data/processed/{dataset_name}.xlsx', index_col=None, header=0)

        if 'cdi_label' in data.columns:
            data = data.rename(columns={'cdi_label': 'outcome'})
        else:
            data = data.rename(columns={'cdi_label_1': 'outcome'})

        # numeric_features = [
        #     'tdeescri', 'tdearitm', 'tdeleit', 'tdetotal', 'age', 'time_in_institution', 'number_siblings', 'people_in_house', 'earnings' 
        # ]
        # numeric_features = [
        #     'tdeescri', 'tdearitm', 'tdeleit', 'tdetotal', 'age' 
        # ]
        # # Standardise the features
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(data[numeric_features])

        # scaled_features_df = pd.DataFrame(scaled_features, columns=numeric_features)

        # # Combine the scaled features with the remaining categorical features
        # data = pd.concat([scaled_features_df, data.drop(columns=numeric_features)], axis=1)

        print("Real data outcome label distribution:")
        print(data['outcome'].value_counts())

        models_to_test = [SparseModelType.NORMAL, SparseModelType.LAPLACE, SparseModelType.CAUCHY, SparseModelType.HORSESHOE, SparseModelType.HORSESHOE_REGULARISED]
        
        results = []
        loo_scores = []
        for model_type in models_to_test:
            model_name = model_type.name.lower()
            print(f"Running model {model_name}")

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

            # Plot the posterior distributions
            plot_posterior_distribution_real(alpha_samples, beta_samples, experiment_name, dataset_name, model_name)

            # Calculate confusion matrix metrics
            X = data.drop('outcome', axis=1).values
            y = data['outcome'].values
            confusion_metrics = calculate_confusion_matrix_metrics(X, y, alpha_samples, beta_samples)

            plot_and_save_roc_curve(X, y, alpha_samples, beta_samples, experiment_name, model_name, experiment_name, dataset_name=dataset_name)

            # Identify active and inactive features (active if 95% CI for beta does not contain zero)
            active_features, inactive_features = identify_active_and_inactive_features(data, beta_samples, experiment_name, model_name, experiment_name, dataset_name=dataset_name)

            results.append({
                "model": model_name,
                **confusion_metrics,
                "active_features": active_features,
                "inactive_features": inactive_features
            })

        os.makedirs(f'sparse_regression/results/experiment_{experiment_name}/{dataset_name}', exist_ok=True)

        loo_scores_df = pd.DataFrame(loo_scores).sort_values(by='elpd_loo', ascending=False)
        loo_scores_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/{dataset_name}/loo_scores.csv', index=False)
        compare_loo_scores_and_save_the_png(loo_scores_df, experiment_name, dataset_name=dataset_name, scenario_name=None)
        
        results_df = pd.DataFrame(results)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.colheader_justify', 'left')
        pd.set_option('display.width', None)

        results_df.to_csv(f'sparse_regression/results/experiment_{experiment_name}/{dataset_name}/evaluation_metrics.csv', index=False)
        print(f'Finished running the experiment {experiment_name} on the dataset {dataset_name}.')