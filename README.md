# Consumption-Wealth Ratio Forecasting (cay_FC and cay_MS)

## Overview

This repository implements the calculation, estimation, in-sample and out-of-sample forecasting of the consumption-wealth ratio using both traditional log-linearization (cay_FC) and Markov-switching (cay_MS) models. 
It includes full empirical replication, rolling window forecasting, and preparation for integration into machine learning pipelines. To evaluate and compare the predictive power of the consumption-wealth ratio under different estimation methods (cay_FC vs. cay_MS) for forecasting quarterly excess returns, and to integrate these forecasts into a broader trading and machine learning framework.

Key Features:

- Log-linear consumption-wealth ratio (cay_FC) implementation
- Markov-switching regime estimation (cay_MS) using Gibbs sampling and MLE
- Rolling window out-of-sample forecasting with and without macro controls
- Forecast regression analysis at horizons 1, 4, and 16 quarters
- Preparation for integration into machine learning ensemble models

## Project Structure

.
- data/                   # Raw and cleaned input datasets
- utils.py                # Python scripts for forecasting, and plotting
- results/                # Output tables and figures from analysis
- models.py               # Python scripts for estimation
- notebook.ipynb          # Jupyter notebooks and demo
- requirements.txt        # Python dependencies
- README.md               # This file


---

## Installation

### Prerequisites

- Python 3.10 or newer
- Recommended: Virtual environment (venv or conda)

### Setup

Clone the repository:

### bash
cd cay-regime-switching

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

## Results Summary

- cay_MS models generally outperform cay_FC in Sharpe ratio, indicating utility for volatility-timed trading strategies.
- Macro controls marginally improve forecast stability.
- Rolling window forecasts remain weak in pure R² terms, suggesting combination with other predictors is necessary for production trading strategies.


## Next Steps
 - Integrate cay_FC and cay_MS signals into machine learning models as features
 - Develop an ensemble trading strategy combining these macro signals with technical indicators
 - Implement real-time data updating pipeline for live predictions
 - Backtest portfolio strategies using forecast-driven position sizing

## Contributing

Contributions are welcome! Please feel free to write to me to open an issue and discuss your ideas.

## Author
Stefan Pilegaard Pedersen
June 2025

## License

This project is licensed under the [MIT License](LICENSE).

