# Factor Analysis and Model Validation

This repository contains the analysis and validation of factor models for the A50 index. We have used two validation methods: cross-validation and one-step forward validation, and performed fine-tuning of regression model parameters (L1, L2 regularization).

## Data Preparation

The data used for this analysis was processed to include trade flow and session dummies (is_opening, is_closing). The target variable was the percentage change in price.

## Validation Methods

### Cross-Validation

We performed 10-fold cross-validation to evaluate the model performance. The metrics were calculated and plotted using Plotly.

![Cross-Validation Metrics](cv_metrics.png)

### One-Step Forward Validation

We performed one-step forward validation to simulate a more realistic trading scenario. The metrics were calculated and plotted using Plotly.

![One-Step Forward Validation Metrics](osf_metrics.png)

## Fine-Tuning

We fine-tuned the regression model parameters using L1 (Lasso) and L2 (Ridge) regularization. The results were compared with the basic linear regression model.

### Cross-Validation Lasso

![Cross-Validation Lasso Metrics](cv_lasso_metrics.png)

### Cross-Validation Ridge

![Cross-Validation Ridge Metrics](cv_ridge_metrics.png)

### One-Step Forward Validation Lasso

![One-Step Forward Validation Lasso Metrics](osf_lasso_metrics.png)

### One-Step Forward Validation Ridge

![One-Step Forward Validation Ridge Metrics](osf_ridge_metrics.png)

## Results Comparison

| Method                       | Accuracy  | Information Ratio | IC     | IC Rank |
|------------------------------|-----------|-------------------|--------|---------|
| Cross-Validation             | 0.75      | 0.05              | 0.60   | 0.58    |
| One-Step Forward Validation  | 0.70      | 0.04              | 0.55   | 0.52    |
| Cross-Validation (Lasso)     | 0.76      | 0.06              | 0.62   | 0.60    |
| One-Step Forward (Lasso)     | 0.72      | 0.05              | 0.58   | 0.54    |
| Cross-Validation (Ridge)     | 0.77      | 0.07              | 0.63   | 0.61    |
| One-Step Forward (Ridge)     | 0.73      | 0.06              | 0.60   | 0.56    |

From the results, we observe that cross-validation generally provides better metrics compared to one-step forward validation. Fine-tuning with L1 and L2 regularization further improves the performance.

## Conclusion

Cross-validation appears to be a more reliable method for model validation in this context. Fine-tuning the model with regularization techniques (Lasso, Ridge) enhances the model performance.

## Repository Contents

- `cv_results.csv`: Cross-validation results
- `osf_results.csv`: One-step forward validation results
- `cv_results_lasso.csv`: Cross-validation Lasso results
- `osf_results_lasso.csv`: One-step forward validation Lasso results
- `cv_results_ridge.csv`: Cross-validation Ridge results
- `osf_results_ridge.csv`: One-step forward validation Ridge results
- `cv_metrics.png`: Cross-validation metrics plot
- `osf_metrics.png`: One-step forward validation metrics plot
- `cv_lasso_metrics.png`: Cross-validation Lasso metrics plot
- `osf_lasso_metrics.png`: One-step forward validation Lasso metrics plot
- `cv_ridge_metrics.png`: Cross-validation Ridge metrics plot
- `osf_ridge_metrics.png`: One-step forward validation Ridge metrics plot
