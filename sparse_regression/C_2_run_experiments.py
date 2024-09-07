import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import os 

from my_thesis.sparse_regression.models.logistic_model_fitting import fit_model, SparseModelType
from my_thesis.sparse_regression.scripts.plot_posterior import  plot_posterior_distribution_real
from my_thesis.sparse_regression.scripts.evaluate import  calculate_confusion_matrix_metrics, identify_active_and_inactive_features, plot_and_save_roc_curve

experiment_name = 'C2'

##### TEST THE MODELS ON A REAL DATA #####
# Load real data without grouping questionnaires by test (with imputed missing values)
data = pd.read_excel('../encoded_imputed_data_T1.xlsx', index_col=None, header=0)
data = data.rename(columns={'cdi_label_1': 'outcome'})

numeric_features = [
    'tdeescri', 'tdearitm', 'tdeleit', 'tdetotal', 'age', 'time_in_institution', 'number_siblings', 'people_in_house', 'earnings' 
]

# Standardise the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[numeric_features])

scaled_features_df = pd.DataFrame(scaled_features, columns=numeric_features)

# Combine the scaled features with the remaining categorical features
data = pd.concat([scaled_features_df, data.drop(columns=numeric_features)], axis=1)

print("Real data outcome label distribution:")
print(data['outcome'].value_counts())

models_to_test = [SparseModelType.NORMAL, SparseModelType.LAPLACE, SparseModelType.CAUCHY, SparseModelType.HORSESHOE, SparseModelType.HORSESHOE_BETTER, SparseModelType.HORSESHOE_REGULARISED, SparseModelType.MIXTURE]

results = []

for model_type in models_to_test:
    model_name = model_type.name.lower()
    print(f"Running model {model_name}")

    fit = fit_model(data, model_type)

    # Extract posterior samples
    beta_samples = fit['beta']
    alpha_samples = fit['alpha']

    # Plot the posterior distributions
    plot_posterior_distribution_real(alpha_samples, beta_samples, experiment_name, 'synthetic_data', model_name)

    # Calculate confusion matrix metrics
    X = data.drop('outcome', axis=1).values
    y = data['outcome'].values
    confusion_metrics = calculate_confusion_matrix_metrics(X, y, alpha_samples, beta_samples)

    plot_and_save_roc_curve(X, y, alpha_samples, beta_samples, experiment_name, 'synthetic_data', model_name)

    # Identify active and inactive features (active if 95% CI for beta does not contain zero)
    active_features, inactive_features = identify_active_and_inactive_features(data, beta_samples, experiment_name, model_name)

    results.append({
        "model": model_name,
        **confusion_metrics,
        "active_features": active_features,
        "inactive_features": inactive_features
    })

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.width', None)

os.makedirs(f'results/experiment_{experiment_name}', exist_ok=True)
results_df.to_csv(f'results/experiment_{experiment_name}/evaluation_metrics.csv', index=False)
print(results_df)