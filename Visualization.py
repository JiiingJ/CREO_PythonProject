from typing import Dict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
from sklearn.utils import resample

from Bootstrapping import add_session_dummies


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


# Function to calculate metrics
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(np.round(y_true), np.round(y_pred))
    ir = np.mean(y_true - y_pred) / np.std(y_true - y_pred)
    ic, _ = pearsonr(y_true, y_pred)
    ic_rank, _ = pearsonr(np.argsort(y_true), np.argsort(y_pred))

    return {
        'accuracy': acc,
        'information_ratio': ir,
        'ic': ic,
        'ic_rank': ic_rank,
    }


# Load your composite data
composite_data = pd.read_csv('path_to_your_data.csv')  # Load your composite data

# Add session dummies and other preprocessing steps
composite_data['Time'] = pd.to_datetime(composite_data['Time'])
composite_data = add_session_dummies(composite_data)
composite_data = composite_data.dropna()

# Perform cross-validation
cv_results = cross_validation(composite_data, 'price_pct_chg', model_type='linear')

# Perform one-step forward validation
osf_results = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='linear')


# Generate plots using Plotly
def plot_metrics(metrics_df: pd.DataFrame, title: str, filename: str):
    fig = px.box(metrics_df, title=title)
    fig.write_image(filename)
    return fig


# Plot cross-validation results
fig_cv = plot_metrics(cv_results, "Cross-Validation Metrics", "cv_metrics.png")

# Plot one-step forward validation results
fig_osf = plot_metrics(osf_results, "One-Step Forward Validation Metrics", "osf_metrics.png")

# Save results to CSV
cv_results.to_csv('cv_results.csv', index=False)
osf_results.to_csv('osf_results.csv', index=False)

# Fine-tuning with L1 and L2 regularization
cv_results_lasso = cross_validation(composite_data, 'price_pct_chg', model_type='lasso')
cv_results_ridge = cross_validation(composite_data, 'price_pct_chg', model_type='ridge')
osf_results_lasso = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='lasso')
osf_results_ridge = one_step_forward_validation(composite_data, 'price_pct_chg', model_type='ridge')

# Plot fine-tuning results
fig_cv_lasso = plot_metrics(cv_results_lasso, "Cross-Validation Lasso Metrics", "cv_lasso_metrics.png")
fig_cv_ridge = plot_metrics(cv_results_ridge, "Cross-Validation Ridge Metrics", "cv_ridge_metrics.png")
fig_osf_lasso = plot_metrics(osf_results_lasso, "One-Step Forward Validation Lasso Metrics", "osf_lasso_metrics.png")
fig_osf_ridge = plot_metrics(osf_results_ridge, "One-Step Forward Validation Ridge Metrics", "osf_ridge_metrics.png")

# Save fine-tuning results to CSV
cv_results_lasso.to_csv('cv_results_lasso.csv', index=False)
cv_results_ridge.to_csv('cv_results_ridge.csv', index=False)
osf_results_lasso.to_csv('osf_results_lasso.csv', index=False)
osf_results_ridge.to_csv('osf_results_ridge.csv', index=False)

# Show plots
fig_cv.show()
fig_osf.show()
fig_cv_lasso.show()
fig_cv_ridge.show()
fig_osf_lasso.show()
fig_osf_ridge.show()
