import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.special import expit as logistic
import pandas as pd
import arviz as az
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

def check_credible_intervals(alpha_samples, beta_samples, alpha_true, betas_true, num_active_features):
    # Calculate 95% credible intervals for alpha and beta
    ci_lower_alpha = np.percentile(alpha_samples, 2.5)
    ci_upper_alpha = np.percentile(alpha_samples, 97.5)
    
    ci_lower_beta = np.percentile(beta_samples, 2.5, axis=1)
    ci_upper_beta = np.percentile(beta_samples, 97.5, axis=1)
    
    # Counters for conditions
    satisfied_alpha = 0
    unsatisfied_alpha = 0

    satisfied_active = 0
    unsatisfied_active = 0

    satisfied_inactive = 0
    unsatisfied_inactive = 0

    # Check alpha
    if ci_lower_alpha <= alpha_true <= ci_upper_alpha:
        satisfied_alpha += 1
    else:
        unsatisfied_alpha += 1

    # Check betas
    for i in range(beta_samples.shape[0]):
        # If beta is active, then check if the true beta lies within the 95% CI
        if i < num_active_features:
            # Active features
            if ci_lower_beta[i] <= betas_true[i] <= ci_upper_beta[i]:
                satisfied_active += 1
            else:
                unsatisfied_active += 1
        else:
            # If beta is inactive, then check if the 95% CI contains zero
            if ci_lower_beta[i] <= 0 <= ci_upper_beta[i]:
                satisfied_inactive += 1
            else:
                unsatisfied_inactive += 1

    return {
        "satisfied_alpha": satisfied_alpha,
        "unsatisfied_alpha": unsatisfied_alpha,
        "satisfied_active(CI_2.5<true_value<CI_97.5)": satisfied_active,
        "unsatisfied_active": unsatisfied_active,
        "satisfied_inactive (CI_2.5<0<CI_97.5)": satisfied_inactive,
        "unsatisfied_inactive": unsatisfied_inactive
    }

def calculate_confusion_matrix_metrics(X, y, alpha_samples, beta_samples):
    # Get the mean values for alpha and beta
    alpha_mean = np.mean(alpha_samples)
    beta_means = np.mean(beta_samples, axis=1)

    # Predict the outcomes using the posterior mean values
    logit_pred = np.dot(X, beta_means) + alpha_mean
    y_pred = (logit_pred > 0).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate accuracy and f1 metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "F1": f1
    }

def plot_and_save_roc_curve(X, y, alpha_samples, beta_samples, experiment_name, model_name, experiment_type, scenario_name=None, dataset_name=None):
    if experiment_type == 'C2':
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', dataset_name, model_name, 'plots')
    else:
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', 'plots', scenario_name, model_name)
    
    # Get the mean values for alpha and beta
    alpha_mean = np.mean(alpha_samples)
    beta_means = np.mean(beta_samples, axis=1)

    # Predict the outcomes using the posterior mean values
    logit_pred = np.dot(X, beta_means) + alpha_mean
    y_pred_prob = logistic(logit_pred)  # Convert logit to probability using scipy.special.expit

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = roc_auc_score(y, y_pred_prob)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Create directory if it doesn't exist
    os.makedirs(base_save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(f'{base_save_dir}/roc_curve.png')
    plt.close()

def plot_active_inactive_features(df, model_name):
    # sort the dataframe by absolute median beta values and then by Active status
    df_sorted = df.sort_values(by='Median_Beta', key=abs, ascending=False)
    df_sorted = df_sorted.sort_values(by='Active', ascending=True, kind='mergesort')[:15]

    # get values for the violin plot
    plot_data = []
    median_data = {}
    for index, row in df_sorted.iterrows():
        median_data[row['Feature']] = np.median(row['CI_95%_Values'])
        for value in row['CI_95%_Values']:
            plot_data.append({
                'Feature': row['Feature'],
                'Beta Value': value,
                'Status': row['Active']
            })

    plot_df = pd.DataFrame(plot_data)

    # Create a violin plot
    fig = px.violin(plot_df, 
                    x='Beta Value', 
                    y='Feature', 
                    color='Status', 
                    orientation='h', 
                    title=f'Ranked Feature Importance with Active/Inactive Classification for {model_name.capitalize()} Model',
                    labels={'Beta Value':'Beta Values', 'Feature':'Feature'},
                    color_discrete_map={'Active': '#81c784', 'Inactive': '#90a4ae'})

    # Clip the violin plot to the range of the data so it doesn't add extra values
    fig.update_traces(spanmode='hard')

    # Add a vertical line at x=0
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")

    return fig

def identify_active_and_inactive_features(data, beta_samples, experiment_name, model_name, experiment_type, scenario_name=None, dataset_name=None):
    if experiment_type in ['C2', 'D', 'E', 'F']:
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', dataset_name, model_name)
    else:
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', 'plots', scenario_name, model_name)
    # Create a directory to save the results
    os.makedirs(base_save_dir, exist_ok=True)
    

    if data.shape[1] != beta_samples.shape[0]:
        data = data.drop('outcome', axis=1)

    # check if the 95% CI contains zero for each beta then it is inactive otherwise active
    ci_lower_beta = np.percentile(beta_samples, 2.5, axis=1)
    ci_upper_beta = np.percentile(beta_samples, 97.5, axis=1)

    # Calculate the mean of the beta samples for each feature
    # beta_means = np.mean(beta_samples, axis=1)

    # Given that posterior distributions for betas are skewed, use median 
    beta_medians = np.median(beta_samples, axis=1)

    active_features = []
    inactive_features = []
    credible_interval_values = []

    for i in range(beta_samples.shape[0]):
        if ci_lower_beta[i] <= 0 <= ci_upper_beta[i]:
            inactive_features.append(i)
        else:
            active_features.append(i)

        # Get all values within the 95% credible interval for each beta 
        values_within_ci = beta_samples[i][(beta_samples[i] >= ci_lower_beta[i]) & (beta_samples[i] <= ci_upper_beta[i])]
        credible_interval_values.append(values_within_ci)
    
    # get the feature names from the data for active and inactive features
    feature_names = data.columns

    df_features = pd.DataFrame({
        "Feature": feature_names,
        "Median_Beta": beta_medians,
        "Active": ["Inactive" if i in inactive_features else "Active" for i in range(beta_samples.shape[0])],
        "CI_95%_Values": credible_interval_values,
        'beta_samples': list(beta_samples),
        'ci_lower_beta': ci_lower_beta,
        'ci_upper_beta': ci_upper_beta
    })

    df_features['Abs_Median_Beta'] = df_features['Median_Beta'].abs()
    df_features = df_features.sort_values(by='Abs_Median_Beta', ascending=False)

    # save the results to an excel file
    df_features.to_excel(os.path.join(base_save_dir, 'features_evaluation.xlsx'), index=False)

    # save the results to a pickle file so that the credible intervals can be loaded later
    df_features.to_pickle(os.path.join(base_save_dir, 'features_evaluation.pkl'))

    fig_active_inactive_features = plot_active_inactive_features(df_features, model_name)
    pio.write_image(fig_active_inactive_features, os.path.join(base_save_dir, 'features_evaluation_plot.png'), width=1000, height=600, scale=2)

    active_features_df = df_features[df_features['Active'] == 'Active']
    inactive_features_df = df_features[df_features['Active'] == 'Inactive']

    return active_features_df, inactive_features_df

def compare_loo_scores_and_save_the_png(df, experiment_name, dataset_name=None, scenario_name=None):
    if experiment_name in ['C2', 'D', 'E', 'F']:
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', dataset_name)
    elif experiment_name == 'B':
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}')
    else: 
        base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', 'plots', scenario_name)

    models = df['Model']
    idatas = df['idata']

    compare_dict = dict(zip(models, idatas))

    # Compare models using LOO and save the plot
    comparison_results = az.compare(compare_dict, method='stacking', ic='loo')
    ax = az.plot_compare(comparison_results)
    ax.figure.savefig(f'{base_save_dir}/loo_comparison.png', bbox_inches='tight', dpi=300)

def plot_total_and_correctly_identified_active_inactive_features(evaluation_metrics, experiment_name):
    scenarios = evaluation_metrics['scenario'].unique()

    for scenario in scenarios:
        scenario_df = evaluation_metrics[evaluation_metrics['scenario'] == scenario]

        models = scenario_df['model'].values
        total_active = scenario_df['satisfied_active(CI_2.5<true_value<CI_97.5)'].values + scenario_df['unsatisfied_active'].values
        identified_active = scenario_df['satisfied_active(CI_2.5<true_value<CI_97.5)'].values
        total_inactive = scenario_df['satisfied_inactive (CI_2.5<0<CI_97.5)'].values + scenario_df['unsatisfied_inactive'].values
        identified_inactive = scenario_df['satisfied_inactive (CI_2.5<0<CI_97.5)'].values

        bar_width = 0.35
        index = np.arange(len(models))

        # create a plot with two rows for active and inactive features 
        fig, axs = plt.subplots(2, 1, figsize=(14, 8))

        # plot total and correctly identified active features
        axs[0].bar(index, total_active, bar_width, label='Total Active Features', color='#aed581')
        axs[0].bar(index + bar_width, identified_active, bar_width, label='Identified Active Features', color='#558b2f')

        # plot total and correctly identified inactive features
        axs[1].bar(index, total_inactive, bar_width, label='Total Inactive Features', color='#90a4ae')
        axs[1].bar(index + bar_width, identified_inactive, bar_width, label='Identified Inactive Features', color='#37474f')

        # add labels and titles
        axs[0].set_ylabel('Number of Active Features')
        axs[0].set_title(f'Active Features in {scenario}')
        axs[1].set_ylabel('Number of Inactive Features')
        axs[1].set_title(f'Inactive Features in {scenario}')

        axs[0].set_ylim(0, 10)
        axs[1].set_ylim(0, 10)
        
        axs[0].set_xticks(index + bar_width / 2)
        axs[0].set_xticklabels(models)
        axs[1].set_xticks(index + bar_width / 2)
        axs[1].set_xticklabels(models)
        
        axs[0].legend()
        axs[1].legend()

        # prevent overlap
        plt.tight_layout()
        plt.savefig(f'sparse_regression/results/experiment_{experiment_name}/plots/{scenario}/active_inactive_features.png')