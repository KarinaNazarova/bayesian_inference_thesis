import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import Lasso

# Load the imputed data for T1
data = pd.read_excel('imputed_data_T1.xlsx', index_col=None, header=0)

X = data.drop('cdi_label', axis=1)
y = data['cdi_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the base model
rf_basic = RandomForestClassifier(random_state=42)

# Define the random grid of hyperparameters
rf_random_grid_search_vals = {
    'n_estimators': np.arange(100, 1001, 100),
    'max_features': ['auto', 'sqrt'],
    'max_depth': np.arange(10, 111, 10),
    'min_samples_split': np.arange(2, 11, 2),
    'min_samples_leaf': np.arange(1, 11, 2),
    'bootstrap': [True, False]
}

# Set up the random grid search
rf_random_grid_search = RandomizedSearchCV(estimator=rf_basic, 
                                           param_distributions=rf_random_grid_search_vals, 
                                           n_iter=75, cv=5, verbose=2, 
                                           random_state=123, n_jobs=-1)

# Fit the model
rf_random_grid_search.fit(X_train, y_train.values.ravel())
print('Random Forest: Random Grid Search Results are:')
print(rf_random_grid_search.best_params_)

# Evaluate the best model on the test data
y_pred = rf_random_grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with best parameters: {accuracy:.2f}")

# Extract the feature importances from the best model
feature_importances = rf_random_grid_search.best_estimator_.feature_importances_

# Get indices of the top 10 important features
indices = np.argsort(feature_importances)[::-1][:10]

# Print the feature ranking
print("Top 10 Feature ranking:")
for i in range(len(indices)):
    print(f"{i+1}. Feature '{X.columns[indices[i]]}' (Importance: {feature_importances[indices[i]]:.4f})")




# Apply Lasso
lasso = Lasso(alpha=0.1)  # Adjust alpha to control the strength of regularization
lasso.fit(X_train, y_train)

# Select features with non-zero coefficients
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected features: {X.columns[selected_features]}")