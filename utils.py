import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import statsmodels.api as sm
import glob
import os




def load_macro_data(start='1952-01-01', end='2025-06-30'):
    """
    Loads and processes macroeconomic data from FRED.

    Returns:
        df (pd.DataFrame): DataFrame with real consumption, labor income, asset wealth, and their logs.
    """
    # Real consumption
    ct = web.DataReader('PCECC96', 'fred', start, end)
    # Labor income
    yt = web.DataReader('WASCUR', 'fred', start, end)
    # Asset wealth
    at = web.DataReader('BOGZ1FL192090005Q', 'fred', start, end)

    # Merge into one dataframe
    df = ct.join(yt).join(at)
    df.columns = ['ct', 'yt', 'at']

    # Drop NA
    df = df.dropna()

    # Log-transform
    df['log_ct'] = np.log(df['ct'])
    df['log_yt'] = np.log(df['yt'])
    df['log_at'] = np.log(df['at'])

    return df

def add_macro_features(df, start='1952-01-01', end='2025-06-30'):
    """
    Adds additional macroeconomic features to the DataFrame:
    - Effective Federal Funds Rate (quarterly)
    - CPI inflation rate (quarterly % change)

    Args:
        df (pd.DataFrame): Main macroeconomic DataFrame.
        start (str): Start date for data retrieval.
        end (str): End date for data retrieval.

    Returns:
        pd.DataFrame: DataFrame with added macro features.
    """
    # Effective Federal Funds Rate
    fedfunds = web.DataReader('FEDFUNDS', 'fred', start, end)
    fedfunds = fedfunds.resample('QE').last()  # Convert to quarterly frequency

    # CPI inflation rate (quarterly % change)
    cpi = web.DataReader('CPIAUCSL', 'fred', start, end)
    cpi_q = cpi.resample('QE').last().pct_change()
    cpi_q.columns = ['CPI_inflation']

    # Unemployment Rate
    # Fetch monthly unemployment rate
    unemployment = web.DataReader('UNRATE', 'fred', start, end)
    unemployment.index = pd.to_datetime(unemployment.index)
    unemployment = unemployment.resample('QE').last()


    # Merge with main df
    df = df.merge(fedfunds, left_index=True, right_index=True, how='left')
    df = df.merge(cpi_q, left_index=True, right_index=True, how='left')
    df = df.merge(unemployment, left_index=True, right_index=True, how='left')

    # Rename columns for clarity
    df.rename(columns={'FEDFUNDS': 'interest_rate', 'UNRATE' :'unemployment'}, inplace=True)
    return df

def load_sp500_excess_returns(start='1985-01-01', end='2025-01-01'):
    """
    Loads S&P 500 data and 3-month Treasury Bill rate, computes quarterly excess returns.

    Returns:
        pd.Series: Quarterly S&P 500 excess returns.
    """
    # Fetch S&P 500 data
    sp500 = yf.download('^GSPC', start=start, end=end, interval='1mo')
    if sp500 is None or sp500.empty:
        raise ValueError("No data found for the specified date range.")
    sp500_q = sp500['Close'].resample('QE').last()
    sp500_returns = sp500_q.pct_change().dropna()
    sp500_returns.columns = ['sp500']

    # Fetch 3-month Treasury Bill rate
    rf = web.DataReader('TB3MS', 'fred', start=start, end=end)
    rf['rf_q'] = ((1 + (rf['TB3MS'] / 100)) ** (1/4)) - 1
    rf_q = rf['rf_q'].resample('QE').mean()

    # Calculate excess returns
    returns = pd.concat([sp500_returns, rf_q], axis=1).dropna()
    returns['excess_ret'] = returns['sp500'] - returns['rf_q']

    # Standardize index to quarterly period end
    returns.index = pd.to_datetime(returns.index).to_period('Q').to_timestamp('Q')

    return returns

def merge_macro_excess_returns(df, returns):
    """
    Aligns macroeconomic data and excess returns to quarterly frequency and merges them.

    Args:
        df (pd.DataFrame): Macroeconomic data.
        returns (pd.Series or pd.DataFrame): Excess returns data.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Convert both indices to quarterly periods, then to timestamps at quarter end
    df.index = pd.to_datetime(df.index).to_period('Q').to_timestamp('Q')
    returns.index = pd.to_datetime(returns.index).to_period('Q').to_timestamp('Q')

    # Re-merge
    merged = df.merge(returns[['excess_ret']], left_index=True, right_index=True, how='inner')
    
    return merged

def create_horizons(df):
    horizons = [1, 4, 16]
    for h in horizons:
        df[f'future_ret_{h}q'] = df['excess_ret'].rolling(window=h).sum().shift(-h+1)
    return horizons

def hamilton_filter(y, X, alpha, beta, sigma2, P, pi0):
    T = len(y)
    k = len(alpha)
    xi = np.zeros((T, k))   # filtered probabilities

    # Step 1: initialization
    ll = np.zeros((T, k))
    for j in range(k):
        mean = alpha[j] + beta * X[0]
        ll[0,j] = norm.pdf(y[0], loc=mean, scale=np.sqrt(sigma2))
    xi[0,:] = pi0 * ll[0,:]
    xi[0,:] /= np.sum(xi[0,:])

    # Step 2: recursion
    for t in range(1,T):
        for j in range(k):
            mean = alpha[j] + beta * X[t]
            ll[t,j] = norm.pdf(y[t], loc=mean, scale=np.sqrt(sigma2))
        xi[t,:] = (xi[t-1,:] @ P) * ll[t,:]
        xi[t,:] /= np.sum(xi[t,:])

    return xi

def backward_sampling(xi, P):
    T, k = xi.shape
    s_t = np.zeros(T, dtype=int)

    # Step 1: sample last state
    s_t[T-1] = np.random.choice(k, p=xi[T-1,:])

    # Step 2: backward sampling
    for t in reversed(range(T-1)):
        prob = xi[t,:] * P[:,s_t[t+1]]
        prob /= np.sum(prob)
        s_t[t] = np.random.choice(k, p=prob)

    return s_t

def cay_MS_MLE_with_macro_forecast(df, horizons, model='yt', macro_controls=['interest_rate', 'CPI_inflation', 'unemployment'], save_csv=True):
    """
    Runs and prints OLS forecasting regressions of future returns on MLE-based CAY variables with macroeconomic controls for multiple horizons.
    Does NOT compare with a no-macro variant.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing future returns, CAY variables, and macroeconomic controls.
    horizons : iterable
        Iterable of forecast horizons (in quarters) to use for the dependent variable (e.g., [1, 4, 8]).
    model : str, optional
        Model type suffix used in column naming (default is 'yt').
    macro_controls : list of str, optional
        List of column names to use as macroeconomic control variables in the regression.
    save_csv : bool, optional
        If True, saves the regression results to a CSV file.

    Returns
    -------
    results_df : pandas.DataFrame or None
        Regression results DataFrame if save_csv is True and results exist, otherwise None.
    """

    results = []

    # Define MLE-based cay_MS variable name
    cay_ms_mle_var = f'cay_MS_MLE_{model}'

    for h in horizons:
        if cay_ms_mle_var not in df.columns:
            print(f"Skipping {cay_ms_mle_var}: not in dataframe.")
            continue

        # Check macro control columns exist
        missing_controls = [col for col in macro_controls if col not in df.columns]
        if missing_controls:
            print(f"Skipping horizon {h} for {cay_ms_mle_var}: missing macro controls {missing_controls}")
            continue

        df_reg = df.dropna(subset=[f'future_ret_{h}q', cay_ms_mle_var, *macro_controls])
        if df_reg.empty:
            print(f"\n🔷 Horizon h={h}q: No data available for {cay_ms_mle_var}.")
            continue

        X = df_reg[[cay_ms_mle_var, *macro_controls]]
        X = sm.add_constant(X)
        y = df_reg[f'future_ret_{h}q']

        model_reg = sm.OLS(y, X).fit()
        print(f"\n🔷 Forecasting regression for horizon h={h}q using {cay_ms_mle_var} (MLE) with macro controls")
        print(model_reg.summary())

        if save_csv:
            for var in [cay_ms_mle_var] + macro_controls:
                results.append({
                    'horizon': h,
                    'cay_var': cay_ms_mle_var,
                    'variable': var,
                    'coef': model_reg.params.get(var, float('nan')),
                    'tstat': model_reg.tvalues.get(var, float('nan')),
                    'pval': model_reg.pvalues.get(var, float('nan')),
                    'r2': model_reg.rsquared,
                    'nobs': int(model_reg.nobs)
                })

    csv_path = f'results/cay_MS_MLE_with_macro_results_{model}.csv'
    if save_csv and results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nRegression results saved to {csv_path}")
        return results_df
    else:
        return None

def compare_cay_FC_vs_MS_with_macro(df, horizons, model = 'yt', macro_controls = ['interest_rate', 'CPI_inflation', 'unemployment'], save_csv=True):
    """
    Runs and prints OLS forecasting regressions of future returns on CAY variables and macroeconomic controls for multiple horizons.
    Optionally saves regression results (coefficients, t-stats, p-values, R2, etc.) to a CSV file.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing future returns, CAY variables, and macroeconomic controls.
    macro_controls : list of str, optional
        List of column names to use as macroeconomic control variables in the regression (default is ['interest_rate', 'CPI_inflation', 'unemployment']).
    horizons : iterable
        Iterable of forecast horizons (in quarters) to use for the dependent variable (e.g., [1, 4, 8]).
    save_csv : bool, optional
        If True, saves the regression results to a CSV file.
    csv_path : str, optional
        Path to save the CSV file if save_csv is True.
    Returns
    -------
    None
        Prints regression summaries to the console. Optionally saves results to CSV.
    """
    results = []
    for h in horizons:
        for cay_var in [f'cay_FC_{model}', f'cay_MS_{model}']:
            if cay_var not in df.columns:
                continue

            df_reg = df.dropna(subset=[f'future_ret_{h}q', cay_var, *macro_controls])
            
            if df_reg.empty:
                print(f"\n🔷 Horizon h={h}q: No data available for {cay_var}.")
                continue
            
            X = df_reg[[cay_var, *macro_controls]]
            X = sm.add_constant(X)
            y = df_reg[f'future_ret_{h}q']
            
            model_reg = sm.OLS(y, X).fit()
            print(f"\n🔷 Forecasting regression for horizon h={h}q using {cay_var} with added macro indicators")
            print(model_reg.summary())

            if save_csv:
                for var in [cay_var] + macro_controls:
                    results.append({
                        'horizon': h,
                        'cay_var': cay_var,
                        'variable': var,
                        'coef': model_reg.params[var],
                        'tstat': model_reg.tvalues[var],
                        'pval': model_reg.pvalues[var],
                        'r2': model_reg.rsquared,
                        'nobs': int(model_reg.nobs)
                    })

    csv_path=f'results/cay_FC_vs_MS_with_macro_results_{model}.csv'
    if save_csv and results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nRegression results saved to {csv_path}")

def compare_cay_FC_vs_MS(df, horizons, model = 'yt', save_csv=True):
    """
    Runs and prints OLS forecasting regressions of future returns on two different CAY variables over multiple horizons.
    For each horizon in `horizons`, this function iterates over the variables 'cay_FC' and 'cay_MS' (if present in the DataFrame),
    and performs an OLS regression of the future return at that horizon on the selected CAY variable. The regression summary is printed
    for each case. If data is missing for a given variable and horizon, a message is printed instead.
    Optionally saves regression results (coefficients, t-stats, p-values, R2, etc.) to a CSV file.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns for future returns (e.g., 'future_ret_1q', 'future_ret_2q', etc.)
        and the CAY variables ('cay_FC', 'cay_MS').
    horizons : iterable
        Iterable of integer horizons (in quarters) to use for forecasting regressions. Each horizon `h` should correspond
        to a column in `df` named 'future_ret_{h}q'.
    save_csv : bool, optional
        If True, saves the regression results to a CSV file.
    Returns
    -------
    None
        This function prints regression summaries and messages to the console, but does not return any value.
    Notes
    -----
    - Requires `statsmodels.api` as `sm` to be imported in the global namespace.
    - Rows with missing data for the dependent or independent variable are dropped before regression.
    """
    results = []
    for h in horizons:
        for cay_var in [f'cay_FC_{model}', f'cay_MS_{model}']:
            if cay_var not in df.columns:
                continue
            
            df_reg = df.dropna(subset=[f'future_ret_{h}q', cay_var])
            
            if df_reg.empty:
                print(f"\n🔷 Horizon h={h}q: No data available for {cay_var}.")
                continue
            
            # Define X and y
            X = df_reg[[cay_var]]
            X = sm.add_constant(X)
            y = df_reg[f'future_ret_{h}q']
            
            # Fit OLS regression
            model_reg = sm.OLS(y, X).fit()
            print(f"\n🔷 Forecasting regression for horizon h={h}q using {cay_var}")
            print(model_reg.summary())

            if save_csv:
                for var in [cay_var]:
                    results.append({
                        'horizon': h,
                        'cay_var': cay_var,
                        'variable': var,
                        'coef': model_reg.params[var],
                        'tstat': model_reg.tvalues[var],
                        'pval': model_reg.pvalues[var],
                        'r2': model_reg.rsquared,
                        'nobs': int(model_reg.nobs)
                    })

    csv_path = f'results/cay_FC_vs_MS_results_{model}.csv'
    if save_csv and results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nRegression results saved to {csv_path}")

def load_all_cay_FC_vs_MS_results(results_dir='results'):
    """
    Loads and concatenates all cay_FC_vs_MS_results_*.csv files in the specified directory.

    Args:
        results_dir (str): Directory containing the result CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all results.
    """
    pattern = os.path.join(results_dir, 'cay_FC_vs_MS_results_*.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    all_results = pd.concat(dfs, ignore_index=True)
    all_results.to_csv(f'results/cay_FC_vs_MS_results_all_results.csv', index=False)
    print(f"Results saved to cay_FC_vs_MS_results_all_results.csv")
    return all_results

def rolling_oos_forecast(df, predictor_col, target_col, min_train=60, horizon=1, freq=4, verbose=True):
    """
    Rolling expanding window out-of-sample forecast evaluation.

    Parameters:
    - df: dataframe with predictor and target
    - predictor_col: string, name of predictor column
    - target_col: string, name of target column (e.g. future_ret_4q)
    - min_train: minimum initial training window size
    - horizon: forecast horizon
    - freq: annualization factor for Sharpe ratio
    - verbose: print progress

    Returns:
    - results_df: dataframe with date, true, pred, residual, oos_r2, implied_sr
    """
    y_true_all = []
    y_pred_all = []

    dates = df.index[min_train + horizon - 1:]  # dates for which forecast is made

    for t in range(min_train, len(df) - horizon + 1):
        train_df = df.iloc[:t]
        test_df = df.iloc[t + horizon - 1]

        # Fit OLS regression (intercept + predictor)
        X_train = np.column_stack((np.ones(len(train_df)), train_df[predictor_col].values))
        y_train = train_df[target_col].values

        beta_hat = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        # Forecast
        x_test = np.array([1, test_df[predictor_col]])
        y_pred = np.dot(x_test, beta_hat)
        y_true = test_df[target_col]

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        if verbose and (t % 20 == 0):
            print(f"Iteration {t}/{len(df) - horizon}")

    # Convert to arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Compute OOS R²
    mse = mean_squared_error(y_true_all, y_pred_all)
    var = np.var(y_true_all, ddof=1)
    oos_r2 = 1 - mse / var

    # Implied Sharpe
    sr_unannualized = np.sqrt(max(oos_r2,0))
    sr_annualized = sr_unannualized * np.sqrt(freq)

    # Results dataframe
    results_df = pd.DataFrame({
        'date': dates,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'residual': y_true_all - y_pred_all
    })
    results_df.set_index('date', inplace=True)

    if verbose:
        print(f"OOS R²: {oos_r2:.4f}")
        print(f"Implied annualized Sharpe ratio: {sr_annualized:.4f}")

    return results_df, oos_r2, sr_annualized

def compute_implied_sharpe(y_true, y_pred, freq=4):
    """
    Computes the implied Sharpe ratio from regression forecast residuals.

    Parameters:
    - y_true: array-like, true realized returns
    - y_pred: array-like, predicted returns from regression
    - freq: int, annualization factor (e.g. 4 for quarterly data)

    Returns:
    - sr_annualized: annualized Sharpe ratio
    """
    # Compute residuals
    residuals = y_true - y_pred

    # Standard deviation of true returns and residuals
    sigma_r = np.std(y_true, ddof=1)
    sigma_e = np.std(residuals, ddof=1)

    # R-squared
    r_squared = 1 - (np.var(residuals, ddof=1) / np.var(y_true, ddof=1))

    # Implied Sharpe ratio (unannualized)
    sr = np.sqrt(max(r_squared,0))

    # Annualize
    sr_annualized = sr * np.sqrt(freq)

    print(f"R²: {r_squared:.4f}")
    print(f"Implied unannualized SR: {sr:.4f}")
    print(f"Implied annualized SR: {sr_annualized:.4f}")

    return sr_annualized

# === Example usage ===

# Replace with your actual regression inputs
# y_true = df['future_ret_4q'].values
# y_pred = regression_model.predict(df[['const', 'cay_FC_yt']])

# sr = compute_implied_sharpe(y_true, y_pred, freq=4)

def forecast_regressions(df, predictor_vars, horizons, macro_controls=None, save_csv=True, csv_prefix='forecast_results'):
    """
    Runs OLS forecasting regressions of future returns on a list of predictor variables with optional macro controls.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing future returns, predictor variables, and macro controls.
    predictor_vars : list of str
        List of predictor variable column names to include as X.
    horizons : iterable
        List of forecast horizons (in quarters) to use for dependent variables (e.g., [1,4,16]).
    macro_controls : list of str, optional
        List of macro control column names to include in regressions.
    save_csv : bool, optional
        If True, saves the regression results to a CSV file.
    csv_prefix : str, optional
        Filename prefix for the saved CSV.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame of regression results.
    """

    results = []

    for h in horizons:
        y_col = f'future_ret_{h}q'

        for var in predictor_vars:
            if var not in df.columns:
                print(f"Skipping {var}: not in dataframe.")
                continue

            required_cols = [y_col, var] + (macro_controls if macro_controls else [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Skipping {var} horizon {h}: missing columns {missing_cols}")
                continue

            df_reg = df.dropna(subset=required_cols)
            if df_reg.empty:
                print(f"\n🔷 Horizon h={h}q: No data available for {var}.")
                continue

            # Define X and y
            X_cols = [var] + (macro_controls if macro_controls else [])
            X = sm.add_constant(df_reg[X_cols])
            y = df_reg[y_col]

            model_reg = sm.OLS(y, X).fit()
            print(f"\n🔷 Forecasting regression for horizon h={h}q using {var} {'with macro controls' if macro_controls else '(no macro)'}")
            print(model_reg.summary())

            # Store results
            for x_var in X_cols:
                results.append({
                    'horizon': h,
                    'predictor': var,
                    'variable': x_var,
                    'coef': model_reg.params.get(x_var, float('nan')),
                    'tstat': model_reg.tvalues.get(x_var, float('nan')),
                    'pval': model_reg.pvalues.get(x_var, float('nan')),
                    'r2': model_reg.rsquared,
                    'nobs': int(model_reg.nobs)
                })

    # Convert to dataframe and save if needed
    results_df = pd.DataFrame(results)

    if save_csv and not results_df.empty:
        csv_path = f'results/{csv_prefix}.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nRegression results saved to {csv_path}")

    return results_df


def backtest_directional_strategy(df, forecast_col, return_col, cost_bp=0, plot=True):
    """
    Backtest a simple directional trading strategy:
    - Long if forecast > 0, Short if forecast < 0
    - Computes strategy returns, cumulative returns, Sharpe, drawdown
    - Includes transaction costs in basis points

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast and realized return columns.
    forecast_col : str
        Name of the forecast column.
    return_col : str
        Name of the realized return column.
    cost_bp : float
        Transaction cost per trade in basis points (default 0).
    plot : bool
        If True, plots cumulative strategy returns.

    Returns
    -------
    results : dict
        Dictionary with strategy metrics and DataFrame with signals and returns.
    """

    data = df[[forecast_col, return_col]].dropna().copy()
    
    # === Generate trading signals based on forecast sign ===
    data['signal'] = np.sign(data[forecast_col])
    
    # === Calculate strategy returns ===
    data['strategy_ret'] = data['signal'] * data[return_col]
    
    # === Apply transaction costs ===
    # Costs applied when signal changes (position turnover)
    data['trade'] = data['signal'].diff().abs()
    data.loc[data.index[0], 'trade'] = np.abs(data.loc[data.index[0], 'signal'])
    data['cost'] = data['trade'] * cost_bp / 10000  # Convert bp to fraction
    data['strategy_ret_net'] = data['strategy_ret'] - data['cost']
    
    # === Calculate performance metrics ===
    cumret = (1 + data['strategy_ret_net']).cumprod()
    strategy_mean = data['strategy_ret_net'].mean()
    strategy_std = data['strategy_ret_net'].std()
    sharpe = strategy_mean / strategy_std * np.sqrt(4)  # Quarterly to annual Sharpe

    # === Calculate max drawdown ===
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # === Print results ===
    print(f"🔷 Strategy Backtest Results ({forecast_col})")
    print(f"Mean Return: {strategy_mean:.4f}")
    print(f"Std Dev: {strategy_std:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # === Plot cumulative returns ===
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(cumret, label='Strategy Cumulative Return')
        plt.title(f"Strategy Cumulative Return ({forecast_col})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

    # === Return results dict ===
    results = {
        'mean_return': strategy_mean,
        'std_dev': strategy_std,
        'sharpe_ann': sharpe,
        'max_drawdown': max_dd,
        'df': data
    }
    
    return results


def rolling_window_forecast(df, predictor, target, window_size=40, expanding=True):
    preds = []
    actuals = []

    for start in range(len(df) - window_size):
        if expanding:
            train = df.iloc[:window_size + start]
        else:
            train = df.iloc[start:start + window_size]
        test = df.iloc[window_size + start]

        # Training data
        X_train = sm.add_constant(train[[predictor]])
        y_train = train[target]
        model = sm.OLS(y_train, X_train).fit()

        # Test data - build as DataFrame with same columns as X_train
        X_test = pd.DataFrame({predictor: [test[predictor]]})
        X_test = sm.add_constant(X_test, has_constant='add')
        X_test = X_test[X_train.columns]  # Ensure same column order

        # Predict
        y_pred = model.predict(X_test).values[0]

        preds.append(y_pred)
        actuals.append(test[target])

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Metrics
    residuals = actuals - preds
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((actuals - np.mean(actuals))**2)
    sharpe = np.mean(preds) / np.std(preds) * np.sqrt(4)

    results_df = pd.DataFrame({
        'predictor': [predictor],
        'R2_OOS': [r2],
        'RMSE': [rmse],
        'Sharpe_ann': [sharpe]
    })

    print(f"\n🔷 Rolling window OOS evaluation for {predictor}:")
    print(f"R² (OOS): {r2:.4f} | RMSE: {rmse:.4f} | Sharpe Ratio (annualized): {sharpe:.4f}")

    return results_df


def batch_rolling_forecasts(df, predictor_vars, target='future_ret_1q', window_size=40, expanding=True, save_csv=True, csv_path='results/rolling_forecast_summary.csv'):
    """
    Runs rolling window forecast for each predictor in predictor_vars and saves combined results to CSV.
    """
    all_results = []

    for predictor in predictor_vars:
        print(f"\n🔄 Running rolling forecast for {predictor}...")
        try:
            res = rolling_window_forecast(df, predictor=predictor, target=target, window_size=window_size, expanding=expanding)
            all_results.append(res)
        except Exception as e:
            print(f"⚠️ Skipping {predictor} due to error: {e}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        if save_csv:
            import os
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"\n✅ Rolling forecast summary saved to {csv_path}")

        return combined_df
    else:
        print("❌ No valid forecasts computed.")
        return pd.DataFrame()
    

def summarize_gibbs_results(df, s_samples_post, alpha_samples, beta_samples, sigma2_samples, model_label='yt'):
    """
    Summarize Gibbs sampler outputs:
    - Plots posterior regime probabilities
    - Returns parameter summary dataframe
    """

    results = {}

    # ---------------------------
    # 1. Calculate posterior regime probabilities
    # ---------------------------
    regime_probs = np.mean(s_samples_post, axis=0)

    # Align regime_probs with df index
    if len(regime_probs) > len(df.index):
        regime_probs_matched = regime_probs[-len(df.index):]
    elif len(regime_probs) < len(df.index):
        raise ValueError("regime_probs length is shorter than df index. Check input alignment.")
    else:
        regime_probs_matched = regime_probs

    # Plot regime probabilities
    plt.figure(figsize=(12,4))
    plt.plot(df.index, regime_probs_matched, label=f'Regime 1 Posterior Probability with {model_label}')
    plt.title(f'Estimated Regime 1 Posterior Probability over Time ({model_label})')
    plt.ylabel('Probability')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save to results dict
    results['regime_probs'] = regime_probs_matched

    # ---------------------------
    # 2. Parameter posterior summaries
    # ---------------------------
    alpha_mean = alpha_samples.mean(axis=0)
    alpha_std = alpha_samples.std(axis=0)
    beta_mean = beta_samples.mean()
    beta_std = beta_samples.std()
    sigma2_mean = sigma2_samples.mean()
    sigma2_std = sigma2_samples.std()

    # Combine into a dataframe
    param_summary = pd.DataFrame({
        'Parameter': [f'alpha_{i}' for i in range(len(alpha_mean))] + ['beta', 'sigma2'],
        'Mean': np.concatenate([alpha_mean, [beta_mean, sigma2_mean]]),
        'Std': np.concatenate([alpha_std, [beta_std, sigma2_std]])
    })

    print("\n🔷 Posterior Parameter Summary:")
    print(param_summary)

    # Save to results dict
    results['param_summary'] = param_summary

    # ---------------------------
    # 3. Effective sample size diagnostics (optional)
    # ---------------------------
    # Example: simple ESS estimate (inefficient but quick)
    def ess(x):
        n = len(x)
        acf_sum = 0
        for lag in range(1, min(1000, n)):
            acf = np.corrcoef(x[:-lag], x[lag:])[0,1]
            if np.isnan(acf) or acf < 0:
                break
            acf_sum += 2 * acf
        return n / (1 + acf_sum)

    beta_ess = ess(beta_samples)
    sigma2_ess = ess(sigma2_samples)

    print(f"\n🔷 Effective Sample Size (ESS): beta = {beta_ess:.1f}, sigma2 = {sigma2_ess:.1f}")

    results['ess_beta'] = beta_ess
    results['ess_sigma2'] = sigma2_ess

    # Return results dictionary
    return results


def rolling_window_forecast_with_macros(df, predictor, target, macro_controls, window_size=40, expanding=True):
    preds, actuals = [], []

    for start in range(len(df) - window_size):
        train = df.iloc[:window_size + start]
        test = df.iloc[window_size + start: window_size + start + 1]

        X_train = train[[predictor] + macro_controls].copy()
        X_train = sm.add_constant(X_train)

        y_train = train[target]

        model = sm.OLS(y_train, X_train).fit()

        X_test = test[[predictor] + macro_controls].copy()
        X_test = sm.add_constant(X_test)

        # === Ensure all train columns exist in test ===
        for col in X_train.columns:
            if col not in X_test.columns:
                if col == 'const':
                    X_test[col] = 1.0
                else:
                    X_test[col] = 0.0

        X_test = X_test[X_train.columns]

        y_pred = model.predict(X_test).values[0]

        preds.append(y_pred)
        actuals.append(test[target].values[0])

    preds = np.array(preds)
    actuals = np.array(actuals)

    r2 = r2_score(actuals, preds)
    returns = preds
    sharpe = (returns.mean() / returns.std()) * np.sqrt(4) if returns.std() > 0 else np.nan

    results_df = pd.DataFrame({'preds': preds, 'actuals': actuals})

    return results_df, r2, sharpe

def batch_rolling_forecasts_with_macros(df, predictor_vars, macro_controls, target='future_ret_1q', window_size=40, expanding=True):
    all_results = []

    for predictor in predictor_vars:
        print(f"\n🔄 Running rolling forecast with macros for {predictor}...")
        try:
            results_df, r2, sharpe = rolling_window_forecast_with_macros(
                df, predictor=predictor, target=target, macro_controls=macro_controls,
                window_size=window_size, expanding=expanding
            )
            all_results.append({
                'predictor': predictor,
                'R2_OOS': r2 if r2 is not None else np.nan,
                'Sharpe_ann': sharpe if sharpe is not None else np.nan
            })
        except Exception as e:
            print(f"⚠️ Skipping {predictor} due to error: {e}")

    results_summary = pd.DataFrame(all_results)

    # Save even if empty to avoid KeyError
    results_summary.to_csv('results/rolling_forecast_with_macros_summary.csv', index=False)
    print("\n✅ Results saved to results/rolling_forecast_with_macros_summary.csv")

    return results_summary

