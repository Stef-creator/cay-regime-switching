import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import yfinance as yf
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
