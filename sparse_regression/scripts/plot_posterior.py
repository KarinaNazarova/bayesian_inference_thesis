import numpy as np
import plotly.graph_objects as go
import os

def plot_posterior_distribution(alpha_samples, beta_samples, alpha_true, betas_true, experiment_name, scenario_name, model_name):
    #os.makedirs(f'plots/experiment_{experiment_name}/{scenario_name}/{model_name}', exist_ok=True)
    base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', 'plots', scenario_name, model_name)

    os.makedirs(base_save_dir, exist_ok=True)
    
    # Plot the distribution of predicted coefficients for ALPHA
    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Histogram(x=alpha_samples[0], nbinsx=50, name=f'alpha', opacity=0.5))

    # add 95% CI for alpha
    ci_lower_alpha = np.percentile(alpha_samples[0], 2.5)
    ci_upper_alpha = np.percentile(alpha_samples[0], 97.5)
    fig_alpha.add_vline(x=ci_lower_alpha, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI lower alpha')
    fig_alpha.add_vline(x=ci_upper_alpha, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI upper alpha')

    # Add true value for alpha as a line 
    fig_alpha.add_vline(x=alpha_true, line=dict(color='teal', width=2), name=f'True alpha')

    # Add annotation for the teal line
    fig_alpha.add_annotation(
        x=alpha_true,
        y=max(np.histogram(alpha_true, bins=50)[0]),
        text=f'True alpha = {alpha_true}',
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        bgcolor="rgba(0,255,255,0.2)"
        )

    fig_alpha.update_layout(
            title=f'Posterior Distribution of Alpha and its True Value',
            barmode='overlay',
            xaxis_title=f'Alpha Value',
            yaxis_title='Frequency'
        )

    # Add annotation for credible intervals
    fig_alpha.add_annotation(
            x=(ci_lower_alpha + ci_upper_alpha) / 2,
            y=max(np.histogram(alpha_samples[0], bins=50)[0]),
            text=f'95% CI: [{ci_lower_alpha:.2f}, {ci_upper_alpha:.2f}]',
            showarrow=False,
            bgcolor="rgba(0,0,255,0.1)"
        )

    #fig_alpha.write_image(f'plots/experiment_{experiment_name}/{scenario_name}/{model_name}/alpha_posterior_distribution.png')
    alpha_posterior_distribution_path = os.path.join(base_save_dir, 'alpha_posterior_distribution.png')
    fig_alpha.write_image(alpha_posterior_distribution_path)


    # Plot the distribution of predicted coefficients for each BETA
    for i in range(beta_samples.shape[0]):
        fig_beta = go.Figure()

        # Calculate 95% CI for beta
        ci_lower_beta = np.percentile(beta_samples[i], 2.5)
        ci_upper_beta = np.percentile(beta_samples[i], 97.5)

        # Show the distribution of betas 
        fig_beta.add_trace(go.Histogram(x=beta_samples[i], nbinsx=50, name=f'beta_{i+1}', opacity=0.5))
        fig_beta.add_vline(x=ci_lower_beta, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI lower beta_{i+1}')
        fig_beta.add_vline(x=ci_upper_beta, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI upper beta_{i+1}')

        # Add annotation for credible intervals
        fig_beta.add_annotation(
            x=(ci_lower_beta + ci_upper_beta) / 2,
            y=max(np.histogram(beta_samples[i], bins=50)[0]),
            text=f'95% CI: [{ci_lower_beta:.2f}, {ci_upper_beta:.2f}]',
            showarrow=False,
            bgcolor="rgba(0,0,255,0.1)"
        )

        # Add true value of beta as a line if the feature is active
        if i < len(betas_true):
            fig_beta.add_vline(x=betas_true[i], line=dict(color='teal', width=2), name=f'True beta_{i+1}')

            # Add annotation for the teal line
            fig_beta.add_annotation(
                x=betas_true[i],
                y=max(np.histogram(beta_samples[i], bins=50)[0]),
                text=f'True beta_{i+1} = {betas_true[i]}',
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-40,
                bgcolor="rgba(0,255,255,0.2)"
            )

            fig_beta.update_layout(
                title=f'Posterior Distribution of Beta_{i+1} (Active)',
                barmode='overlay',
                xaxis_title=f'Beta_{i+1} Value',
                yaxis_title='Frequency'
            )
        else: 
            fig_beta.update_layout(
                title=f'Posterior Distribution of Beta_{i+1} (Inactive)',
                barmode='overlay',
                xaxis_title=f'Beta_{i+1} Value',
                yaxis_title='Frequency'
            )

        
        #fig_beta.write_image(f'plots/experiment_{experiment_name}/{scenario_name}/{model_name}/beta_{i+1}_posterior_distribution.png')
        beta_posterior_distribution_path = os.path.join(base_save_dir, f'beta_{i+1}_posterior_distribution.png')
        fig_beta.write_image(beta_posterior_distribution_path)


def plot_posterior_distribution_real(alpha_samples, beta_samples, experiment_name, scenario_name, model_name):
    # os.makedirs(f'plots/experiment_{experiment_name}/{scenario_name}/{model_name}', exist_ok=True)
    base_save_dir = os.path.join('sparse_regression', 'results', f'experiment_{experiment_name}', scenario_name, model_name, 'plots')

    os.makedirs(base_save_dir, exist_ok=True)
    
    # Calculate mean for alpha and beta
    alpha_mean = np.round(np.mean(alpha_samples), 2)
    beta_means = np.round(np.mean(beta_samples.T, axis=0), 2)
    
    # Plot the distribution of predicted coefficients for ALPHA
    fig_alpha = go.Figure()
    fig_alpha.add_trace(go.Histogram(x=alpha_samples[0], nbinsx=50, name=f'alpha', opacity=0.5))

    # add 95% CI for alpha
    ci_lower_alpha = np.percentile(alpha_samples[0], 2.5)
    ci_upper_alpha = np.percentile(alpha_samples[0], 97.5)
    fig_alpha.add_vline(x=ci_lower_alpha, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI lower alpha')
    fig_alpha.add_vline(x=ci_upper_alpha, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI upper alpha')

    # Add mean value for alpha as a line 
    fig_alpha.add_vline(x=alpha_mean, line=dict(color='teal', width=2), name=f'True alpha')

    # Add annotation for the teal line
    fig_alpha.add_annotation(
        x=alpha_mean,
        y=max(np.histogram(alpha_mean, bins=50)[0]),
        text=f'Mean alpha = {alpha_mean}',
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        bgcolor="rgba(0,255,255,0.2)"
        )

    fig_alpha.update_layout(
            title=f'Posterior Distribution of Alpha and its Mean Value',
            barmode='overlay',
            xaxis_title=f'Alpha Value',
            yaxis_title='Frequency'
        )

    # Add annotation for credible intervals
    fig_alpha.add_annotation(
            x=(ci_lower_alpha + ci_upper_alpha) / 2,
            y=max(np.histogram(alpha_samples[0], bins=50)[0]),
            text=f'95% CI: [{ci_lower_alpha:.2f}, {ci_upper_alpha:.2f}]',
            showarrow=False,
            bgcolor="rgba(0,0,255,0.1)"
        )

    fig_alpha.write_image(f'{base_save_dir}/alpha_posterior_distribution.png')

    # Plot the distribution of predicted coefficients for each BETA
    for i in range(beta_samples.shape[0]):
        fig_beta = go.Figure()

        # Calculate 95% CI for beta
        ci_lower_beta = np.percentile(beta_samples[i], 2.5)
        ci_upper_beta = np.percentile(beta_samples[i], 97.5)

        # Show the distribution of betas 
        fig_beta.add_trace(go.Histogram(x=beta_samples[i], nbinsx=50, name=f'beta_{i+1}', opacity=0.5))
        fig_beta.add_vline(x=ci_lower_beta, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI lower beta_{i+1}')
        fig_beta.add_vline(x=ci_upper_beta, line=dict(color='darkblue', width=2, dash='dash'), name=f'95% CI upper beta_{i+1}')

        # Add annotation for credible intervals
        fig_beta.add_annotation(
            x=(ci_lower_beta + ci_upper_beta) / 2,
            y=max(np.histogram(beta_samples[i], bins=50)[0]),
            text=f'95% CI: [{ci_lower_beta:.2f}, {ci_upper_beta:.2f}]',
            showarrow=False,
            bgcolor="rgba(0,0,255,0.1)"
        )

        # Add true value of beta as a line if the feature is active
        if i < len(beta_means):
            fig_beta.add_vline(x=beta_means[i], line=dict(color='teal', width=2), name=f'True beta_{i+1}')

            # Add annotation for the teal line
            fig_beta.add_annotation(
                x=beta_means[i],
                y=max(np.histogram(beta_samples[i], bins=50)[0]),
                text=f'True beta_{i+1} = {beta_means[i]}',
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-40,
                bgcolor="rgba(0,255,255,0.2)"
            )

            fig_beta.update_layout(
                title=f'Posterior Distribution of Beta_{i+1} (Active)',
                barmode='overlay',
                xaxis_title=f'Beta_{i+1} Value',
                yaxis_title='Frequency'
            )
        else: 
            fig_beta.update_layout(
                title=f'Posterior Distribution of Beta_{i+1} (Inactive)',
                barmode='overlay',
                xaxis_title=f'Beta_{i+1} Value',
                yaxis_title='Frequency'
            )

        
        fig_beta.write_image(f'{base_save_dir}/beta_{i+1}_posterior_distribution.png')
