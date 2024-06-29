# Market Data Analysis and Factor Model Validation Framework

This repository combines two main branches: Market Data Analysis and Backtesting Framework, and Factor Analysis and Model Validation. Below, you will find the key components and functionalities of the combined repository.

## Market Data Analysis and Backtesting Framework

This section provides a framework for analyzing market data and performing backtesting using various custom-designed factors. The key components of this framework include market data loading, monitoring, and analysis, with support for visualizing results using Plotly.

### Key Components

1. **Market Data Loading**
   - **api.py**: Defines the API for loading market data and adding market data monitors.

2. **Market Data Replay**
   - **utils.py**: Contains the `ProgressiveReplay` class for replaying market data.

3. **Descriptive Statistics**
   - **factor1.py**: Defines the `AggregatedTrade` class for monitoring and aggregating trade data.
   - **backtest1.py**: Sets up a backtest using the `AggregatedTrade` monitor.


4. **Factor Design**
   - **sampler.py**: Helper module for sampling data.
   - **factordesign2.py**: Defines the `ChipDistribution` class for calculating and plotting chip distributions.
   - **backtest factor design.py**: Sets up a backtest using the `ChipDistribution` monitor and calculates prediction power.


## Factor Analysis and Model Validation

This section contains the analysis and validation of factor models for the A50 index. We have used two validation methods: cross-validation and one-step forward validation, and performed fine-tuning of regression model parameters (L1, L2 regularization).

### Data Preparation

The data used for this analysis was processed to include trade flow and session dummies (is_opening, is_closing). The target variable was the percentage change in price.

### Validation Methods

1. **Cross-Validation**
   - Performed 10-fold cross-validation to evaluate the model performance.
   - The metrics were calculated and plotted using Plotly.

2. **One-Step Forward Validation**
   - Simulated a more realistic trading scenario by updating the model with each new data point.


### Regularization Techniques

- **Lasso (L1) Regularization**
  - Applied to enhance model performance.


- **Ridge (L2) Regularization**
  - Applied to enhance model performance.


