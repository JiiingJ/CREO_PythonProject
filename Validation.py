from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np
import pandas as pd

from Bootstrapping import add_session_dummies, calculate_metrics


# Function to perform cross-validation
def cross_validation(data: pd.DataFrame, target: str, model_type='linear', n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        X_train = train_data[['trade_flow', 'is_opening', 'is_closing']]
        y_train = train_data[target]
        X_test = test_data[['trade_flow', 'is_opening', 'is_closing']]
        y_test = test_data[target]

        if model_type == 'lasso':
            model = Lasso()
        elif model_type == 'ridge':
            model = Ridge()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


# Example usage
composite_data = pd.read_csv('path_to_your_data.csv')  # Load your composite data

# Add session dummies and other preprocessing steps
composite_data['Time'] = pd.to_datetime(composite_data['Time'])
composite_data = add_session_dummies(composite_data)
composite_data = composite_data.dropna()

cv_results = cross_validation(composite_data, 'price_pct_chg', model_type='linear')
print(cv_results.describe())


# Function to perform one-step forward validation
def one_step_forward_validation(data: pd.DataFrame, target: str, model_type='linear'):
    metrics_list = []

    for i in range(1, len(data)):
        train_data = data.iloc[:i]
        test_data = data.iloc[i:i + 1]

        X_train = train_data[['trade_flow', 'is_opening', 'is_closing']]
        y_train = train_data[target]
        X_test = test_data[['trade_flow', 'is_opening', 'is_closing']]
        y_test = test_data[target]

        if model_type == 'lasso':
            model = Lasso()
        elif model_type == 'ridge':
            model = Ridge()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


# Example usage
osf_results = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='linear')
print(osf_results.describe())



# Compare cross-validation and one-step forward validation
cv_mean_accuracy = cv_results['accuracy'].mean()
osf_mean_accuracy = osf_results['accuracy'].mean()

print(f'CV Mean Accuracy: {cv_mean_accuracy}')
print(f'OSF Mean Accuracy: {osf_mean_accuracy}')

# Fine-tuning with L1 and L2 regularization
cv_results_lasso = cross_validation(composite_data, 'price_pct_chg', model_type='lasso')
cv_results_ridge = cross_validation(composite_data, 'price_pct_chg', model_type='ridge')
osf_results_lasso = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='lasso')
osf_results_ridge = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='ridge')

# Compare results with regularization
print('Lasso CV Results:', cv_results_lasso.describe())
print('Ridge CV Results:', cv_results_ridge.describe())
print('Lasso OSF Results:', osf_results_lasso.describe())
print('Ridge OSF Results:', osf_results_ridge.describe())
