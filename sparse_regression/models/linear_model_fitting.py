import stan
import enum
import numpy as np

###### STAN MODEL FOR NORMAL SPARSE REGRESSION ######
normal_sparse_regression_model = """
data {
  int<lower=1> N;          // number of observations
  int<lower=0> K;          // number of features
  matrix[N, K] X;          // predictor matrix
  vector[N] y;             // continuous outcome variable
}
parameters {
  vector[K] beta;          // coefficients for predictors
  real alpha;              // intercept
  real<lower=0> sigma;     // noise standard deviation
}
model {
  // Priors
  beta ~ normal(0, 10);
  alpha ~ normal(0, 10);
  sigma ~ normal(0, 5);

  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}
generated quantities {
  vector[N] y_new;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_new[n] = normal_rng(X[n] * beta + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | X[n] * beta + alpha, sigma);
  }
}
"""

###### END STAN MODEL FOR NORMAL SPARSE REGRESSION ######



###### STAN MODEL FOR LAPLACE SPARSE REGRESSION ######
laplace_sparse_regression_model = """
data {
  int<lower=1> N;          // number of observations
  int<lower=0> K;          // number of features
  matrix[N, K] X;          // predictor matrix
  vector[N] y;             // continuous outcome variable
}
parameters {
  vector[K] beta;          // coefficients for predictors
  real alpha;              // intercept
  real<lower=0> tau;       // scale parameter for Laplace distribution
  real<lower=0> sigma;     // noise standard deviation
}
model {
  // Priors
  beta ~ double_exponential(0, tau);
  alpha ~ normal(0, 10);
  sigma ~ normal(0, 5);
  tau ~ normal(0, 1);

  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}
generated quantities {
  vector[N] y_new;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_new[n] = normal_rng(X[n] * beta + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | X[n] * beta + alpha, sigma);
  }
}
"""

###### END STAN MODEL FOR LAPLACE SPARSE REGRESSION ######


###### STAN MODEL FOR CAUCHY SPARSE REGRESSION ######
cauchy_sparse_regression_model = """
data {
  int<lower=1> N;          // number of observations
  int<lower=0> K;          // number of features
  matrix[N, K] X;          // predictor matrix
  vector[N] y;             // continuous outcome variable
}
parameters {
  vector[K] beta;          // coefficients for predictors
  real alpha;              // intercept
  real<lower=0> tau;       // scale parameter for Cauchy distribution
  real<lower=0> sigma;     // noise standard deviation
}
model {
  // Priors
  tau ~ normal(0, 1);
  beta ~ cauchy(0, tau);
  alpha ~ normal(0, 10);

  sigma ~ normal(0, 5);

  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}
generated quantities {
  vector[N] y_new;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_new[n] = normal_rng(X[n] * beta + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | X[n] * beta + alpha, sigma);
  }
}
"""
###### END STAN MODEL FOR CAUCHY SPARSE REGRESSION ######


###### STAN MODEL FOR HORSESHOE SPARSE REGRESSION ######
horseshoe_sparse_regression_model = """
data {
  int<lower=1> N;          // number of observations
  int<lower=0> K;          // number of features
  matrix[N, K] X;          // predictor matrix
  vector[N] y;             // continuous outcome variable
}
parameters {
  vector[K] beta;          // coefficients for predictors
  real alpha;              // intercept
  real<lower=0> tau;       // global shrinkage parameter
  vector<lower=0>[K] lambda; // local shrinkage parameters

  real<lower=0> sigma;     // noise standard deviation
}
transformed parameters {
  vector[K] lambda_hat; // scale parameter for each coefficient
  lambda_hat = lambda * tau;
}
model {
  // Priors
  tau ~ normal(0, 1);  // global shrinkage prior
  lambda ~ cauchy(0, 1);  // local shrinkage priors

  beta ~ normal(0, lambda_hat);
  alpha ~ normal(0, 10);
  sigma ~ normal(0, 5); 
    
  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}
generated quantities {
  vector[N] y_new;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_new[n] = normal_rng(X[n] * beta + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | X[n] * beta + alpha, sigma);
  }
}
"""
###### END STAN MODEL FOR HORSESHOE SPARSE REGRESSION ######

# ###### BETTER STAN MODEL FOR HORSESHOE SPARSE REGRESSION ######
# horseshoe_sparse_logistic_regression_model_better = """
# data {
#     int <lower=0> N; // number of observations
#     int <lower=0> K; // number of predictors
#     matrix [N, K] X; // inputs
#     array[N] int <lower=0, upper=1> y; // outputs

#     real <lower=0> scale_icept; // prior std for the intercept
#     real <lower=0> scale_global; // scale for the half -t prior for tau
#     real <lower=1> nu_global; // degrees of freedom for the half -t prior

#     // for tau
#     real <lower=1> nu_local; // degrees of freedom for the half - t priors
#     }
# parameters {
#     real alpha;
#     vector [K] z; // Base for priors 
#     real <lower=0> tau; // global shrinkage parameter
#     vector <lower=0>[K] lambda; // local shrinkage parameter
# }
# transformed parameters {
#     vector [K] beta ; // regression coefficients
#     vector [N] f;  // latent function values

#     beta = z .* lambda * tau;
#     f = alpha + X*beta;
# }

# model {
# // half -t priors for lambdas and tau , and inverse - gamma for c ^2
#     z ~ normal(0, 1);
#     lambda ~ student_t(nu_local, 0, 1);
#     tau ~ student_t(nu_global, 0, scale_global);
#     alpha ~ normal (0, scale_icept);
    
#     y~bernoulli_logit(f);
# }
# generated quantities {
#   array[N] real log_lik;
#   for (n in 1:N) {
#     log_lik[n] = bernoulli_logit_lpmf(y[n] | dot_product(X[n], beta) + alpha);
#   }
# }
# """
# ###### END BETTER STAN MODEL FOR HORSESHOE SPARSE REGRESSION ######

###### REGULARISED HORSESHOE STAN MODEL FOR SPARSE REGRESSION ######
horseshoe_sparse_regression_model_regularised = """
data {
  int<lower=1> N;          // number of observations
  int<lower=0> K;          // number of features
  matrix[N, K] X;          // predictor matrix
  vector[N] y;             // continuous outcome variable
  real<lower=0> scale_icept; // prior std for the intercept
  real<lower=0> scale_global; // scale for the half-t prior for tau
  real<lower=1> nu_global;   // degrees of freedom for the half-t prior for tau
  real<lower=1> nu_local;    // degrees of freedom for the half-t priors for lambdas
  real<lower=0> slab_scale;  // slab scale for the regularized horseshoe
  real<lower=0> slab_df;     // slab degrees of freedom for the regularized horseshoe
}
parameters {
  real alpha;
  vector[K] z;
  real<lower=0> tau;        // global shrinkage parameter
  vector<lower=0>[K] lambda; // local shrinkage parameter
  real<lower=0> caux;
  real<lower=0> sigma;      // noise standard deviation
}
transformed parameters {
  vector<lower=0>[K] lambda_tilde; // truncated local shrinkage parameter
  real<lower=0> c;               // slab scale
  vector[K] beta;                // regression coefficients
  vector[N] f;                   // latent function values

  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt(c^2 * square(lambda) ./ (c^2 + tau^2 * square(lambda))); 
  beta = z .* lambda_tilde * tau;
  f = alpha + X * beta;
}
model {
  // Priors
  z ~ normal(0, 1);
  lambda ~ student_t(nu_local, 0, 1);
  tau ~ student_t(nu_global, 0, scale_global * sigma);
  caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  alpha ~ normal(0, scale_icept);
  sigma ~ normal(0, 5);
  
  // Likelihood
  y ~ normal(f, sigma);
}
generated quantities {
  vector[N] y_new;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_new[n] = normal_rng(f[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | f[n], sigma);
  }
}
"""
###### END REGULARISED HORSESHOE STAN MODEL FOR SPARSE REGRESSION ######

# ###### STAN MODEL WITH NORMAL MIXTURE SPARSE REGRESSION ######
# mixture_sparse_logistic_regression_model = """
# data {
#   int<lower=1> N;          // number of observations
#   int<lower=0> K;          // number of features
#   matrix[N, K] X;          // predictor matrix
#   array[N] real y;          // binary outcome variable
#   int<lower=1> G;          // number of mixture components
# }
# parameters {
#   simplex[G] theta;        // mixture proportions
#   matrix[K, G] beta_unweighted;       // coefficients for predictors for each mixture component
#   vector[G] alpha_unweighted;         // intercept for each mixture component

#   // Parameters for the normal distributions of beta
#   matrix[K, G] mu_beta;    // mean of normal distributions for beta
#   matrix<lower=0>[K, G] sigma_beta; // standard deviation of normal distributions for beta

#   // Parameters for the normal distributions of alpha
#   vector[G] mu_alpha;        // mean of normal distributions for alpha
#   vector<lower=0>[G] sigma_alpha; // standard deviation of normal distributions for alpha
# }
# model {
#   vector[G] log_theta = log(theta);  // log of mixture proportions for numerical stability

#   // Priors for the means and standard deviations of the normal distributions.
#   // We have each alpha normally distributed for each component, and then the whole distribution of parameters normally distributed for each component 
#   for (g in 1:G) {
#     mu_beta[, g] ~ normal(0, 10);
#     sigma_beta[, g] ~ normal(0, 10);
#     mu_alpha[g] ~ normal(0, 10);
#     sigma_alpha[g] ~ normal(0, 10);
#   }

#   // Priors for the mixture components
#   for (g in 1:G) {
#     beta_unweighted[, g] ~ normal(mu_beta[, g], sigma_beta[, g]);
#     alpha_unweighted[g] ~ normal(mu_alpha[g], sigma_alpha[g]);
#   }

#   // Likelihood
#   for (n in 1:N) {
#     vector[G] log_lik;
#     for (g in 1:G) {
#       log_lik[g] = log_theta[g] + bernoulli_logit_lpmf(y[n] | dot_product(X[n], beta_unweighted[, g]) + alpha_unweighted[g]);
#     }
#     target += log_sum_exp(log_lik);
#   }
# }
# generated quantities {
#   array[N] int y_new;
  
#   // values weighted by the mixture proportions
#   vector[K] beta;
#   real alpha;

#   // Calculate the weighted average of beta coefficients using the mixture proportions
#   for (k in 1:K) {
#     beta[k] = dot_product(theta, beta_unweighted[k,]);
#   }

#   // Calculate the weighted average of alpha using the mixture proportions
#   alpha = dot_product(theta, alpha_unweighted);

#   for (n in 1:N) {
#     real logit_p = alpha + dot_product(X[n], beta);
#     y_new[n] = bernoulli_rng(inv_logit(logit_p)); // use weighted alpha and beta for new predictions
#   }
# }
# """
# ###### END STAN MODEL WITH NORMAL MIXTURE SPARSE REGRESSION ######

class SparseModelType(enum.Enum):
    NORMAL = 'normal'
    LAPLACE = 'laplace'
    CAUCHY = 'cauchy'
    HORSESHOE = 'horseshoe'

    HORSESHOE_BETTER = 'horseshoe_better'
    HORSESHOE_REGULARISED = 'horseshoe_regularised'
    MIXTURE = 'mixture'

def get_model_code(model_type):
    if model_type == SparseModelType.NORMAL:
        return normal_sparse_regression_model
    elif model_type == SparseModelType.LAPLACE:
        return laplace_sparse_regression_model
    elif model_type == SparseModelType.CAUCHY:
        return cauchy_sparse_regression_model
    elif model_type == SparseModelType.HORSESHOE:
        return horseshoe_sparse_regression_model
    
    elif model_type == SparseModelType.HORSESHOE_BETTER:
        return horseshoe_sparse_logistic_regression_model_better
    elif model_type == SparseModelType.HORSESHOE_REGULARISED:
        return horseshoe_sparse_regression_model_regularised
    elif model_type == SparseModelType.MIXTURE:
        return mixture_sparse_logistic_regression_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def fit_model(data, model_type, chains=4, samples=1000):
    model_code = get_model_code(model_type)
    
    # Prepare data for Stan
    X = data.drop('outcome', axis=1).values
    y = data['outcome'].values

    if model_type == SparseModelType.HORSESHOE_REGULARISED:
        model_data = {
          'N': X.shape[0],
          'K': X.shape[1],
          'X': X,
          'y': y,
          'scale_icept': 5,  # (paper) prior std for the intercept: larger => intercept can take more values: https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/4366de036cc608c942fdebb930e96f2cc8b83d71/WIP/Evaluating%20Variational%20Inference/Figure_7_Horseshoe.R
          'scale_global':  (3 / (X.shape[1] - 3)) * (1 / np.sqrt(X.shape[0])), #(formula: (1/(d-1)) * (1/np.sqrt(n)), d-columns, n-rows) smaller value => more shrinkage towards 0
          'nu_global': 20, # dof for the half-t prior for tau: smaller => heavier tails, more deviations from 0 (like cauchy), bigger -> more concentration around the mean (likr normal)
          'nu_local': 10, # smaller -> heavier tails 
          'slab_scale': 2, # (paper) regularisation to prevent extremely large coefficients. larger -> allows larger coefficients 
          'slab_df': 4  #(paper) lower => heavier tails and larger coefficients 
        }
    elif model_type == SparseModelType.HORSESHOE_BETTER:
        model_data = {
          'N': X.shape[0],
          'K': X.shape[1],
          'X': X,
          'y': y,
          'scale_icept': 5,  # (paper) prior std for the intercept: larger => intercept can take more values: https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/4366de036cc608c942fdebb930e96f2cc8b83d71/WIP/Evaluating%20Variational%20Inference/Figure_7_Horseshoe.R
          'scale_global': (1 / (X.shape[1] - 1)) * (2 / np.sqrt(X.shape[0])), #(formula) smaller value => more shrinkage towards 0
          'nu_global': 20, # dof for the half-t prior for tau: smaller => heavier tails, more deviations from 0, bigger -> more concentration around the mean 
          'nu_local': 10 # smaller -> heavier tails 
        }
    elif model_type == SparseModelType.MIXTURE:
        model_data = {
          'N': X.shape[0],
          'K': X.shape[1],
          'X': X,
          'y': y,
          'G': 4
          }
    else:
      model_data = {
          'N': X.shape[0],
          'K': X.shape[1],
          'X': X,
          'y': y
          }
    
    posterior = stan.build(model_code, data=model_data)
    fit = posterior.sample(num_chains=chains, num_samples=samples)



    return fit