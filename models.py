import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from utils import hamilton_filter, backward_sampling
import statsmodels.api as sm
from scipy.stats import invgamma, norm
from numpy.linalg import inv
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

def compute_pca(df):
    """
    Computes the first principal component of log_yt and log_at as 'pca' in df.
    Standardizes the resulting column (mean=0, std=1).
    """
    # Drop rows with NaNs in log_yt or log_at before PCA
    resource_vars = df[['log_yt', 'log_at']].dropna()

    # Fit PCA with 1 component
    pca = PCA(n_components=1)
    lifetime_resource_factor = pca.fit_transform(resource_vars)

    # The PCA output is an array; align it back to df index
    df.loc[resource_vars.index, 'pca'] = lifetime_resource_factor.flatten()

    # Standardize (mean=0, std=1)
    df['pca'] = (df['pca'] - df['pca'].mean()) / df['pca'].std()

    return df

def estimate_cay_MS_via_gibbs_final(df, k_regimes=2, n_iter=1000, burn_in=200, model='yt', verbose=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    """
    Estimate cay_MS using a Markov-switching regression with Gibbs sampling.
    Now with regime-dependent slopes and flatter transition priors.
    """

    # ---------------------------
    # 0. Data preparation
    # ---------------------------
    if model == 'yt':
        y = df['log_ct'].values
        X = df['log_yt'].values
    elif model == 'pca':
        y = df['log_ct'].values
        X = df['pca'].values
    else:
        raise ValueError(f"Unknown model type: {model}")

    T = len(y)

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    sigma2_y = np.var(y)
    mu_alpha = np.zeros(k_regimes)
    sigma2_alpha = np.ones(k_regimes) * sigma2_y
    mu_beta = np.zeros(k_regimes)
    sigma2_beta = np.ones(k_regimes) * sigma2_y
    alpha_sigma = 2.5
    beta_sigma = 0.5
    prior_trans = np.ones((k_regimes, k_regimes)) * 0.5  # Beta(0.5,0.5)-equivalent

    # ---------------------------
    # 2. Initialize storage
    # ---------------------------
    alpha_samples = np.zeros((n_iter, k_regimes))
    beta_samples = np.zeros((n_iter, k_regimes))
    sigma2_samples = np.zeros(n_iter)
    s_samples = np.zeros((n_iter, T))

    # Initialize parameters
    alpha = np.zeros(k_regimes)
    beta = np.zeros(k_regimes)
    sigma2 = 1
    P = np.full((k_regimes, k_regimes), 1/k_regimes)
    s_t = np.random.choice(k_regimes, size=T)

    # ---------------------------
    # Helper functions
    # ---------------------------
    def hamilton_filter(y, X, alpha, beta, sigma2, P, pi0):
        T = len(y)
        k = len(alpha)
        xi = np.zeros((T, k))
        for s in range(k):
            mu = alpha[s] + beta[s] * X[0]
            xi[0, s] = pi0[s] * norm.pdf(y[0], loc=mu, scale=np.sqrt(sigma2))
        xi[0, :] /= xi[0, :].sum()
        for t in range(1, T):
            for s in range(k):
                mu = alpha[s] + beta[s] * X[t]
                xi[t, s] = norm.pdf(y[t], loc=mu, scale=np.sqrt(sigma2)) * np.dot(xi[t-1, :], P[:, s])
            xi[t, :] /= xi[t, :].sum()
        return xi

    def backward_sampling(xi, P):
        T, k = xi.shape
        s_t = np.zeros(T, dtype=int)
        s_t[T-1] = np.random.choice(k, p=xi[T-1, :])
        for t in range(T-2, -1, -1):
            prob = xi[t, :] * P[:, s_t[t+1]]
            prob /= prob.sum()
            s_t[t] = np.random.choice(k, p=prob)
        return s_t

    # ---------------------------
    # 5. Gibbs sampler loop
    # ---------------------------
    for it in range(n_iter):
        # Step 1. Sample s_t using Hamilton filter + backward sampling
        pi0 = np.full(k_regimes, 1/k_regimes)
        xi = hamilton_filter(y, X, alpha, beta, sigma2, P, pi0)
        s_t = backward_sampling(xi, P)

        # Step 2. Sample alpha_s and beta_s with ordering constraint
        for s in range(k_regimes):
            idx = (s_t == s)
            n_s = np.sum(idx)
            if n_s > 0:
                y_s = y[idx]
                X_s = X[idx]

                # Sample beta_s
                var_beta = 1 / (np.sum(X_s**2) / sigma2 + 1 / sigma2_beta[s])
                var_beta = max(var_beta, 1e-8)
                mean_beta = var_beta * (np.sum(X_s * (y_s - alpha[s])) / sigma2)
                beta[s] = norm.rvs(loc=mean_beta, scale=np.sqrt(var_beta))

                # Sample alpha_s
                var_alpha = 1 / (n_s / sigma2 + 1 / sigma2_alpha[s])
                var_alpha = max(var_alpha, 1e-8)
                mean_alpha = var_alpha * (np.sum(y_s - beta[s] * X_s) / sigma2)
                alpha[s] = norm.rvs(loc=mean_alpha, scale=np.sqrt(var_alpha))
            else:
                beta[s] = norm.rvs(loc=mu_beta[s], scale=np.sqrt(sigma2_beta[s]))
                alpha[s] = norm.rvs(loc=mu_alpha[s], scale=np.sqrt(sigma2_alpha[s]))

        # Enforce ordering constraint (α_0 < α_1)
        sort_idx = np.argsort(alpha)
        alpha = alpha[sort_idx]
        beta = beta[sort_idx]
        new_s_t = np.zeros_like(s_t)
        for new_label, old_label in enumerate(sort_idx):
            new_s_t[s_t == old_label] = new_label
        s_t = new_s_t

        # Step 3. Sample sigma2
        resid = y - (alpha[s_t] + beta[s_t] * X)
        alpha_post = alpha_sigma + T/2
        beta_post = beta_sigma + 0.5 * np.sum(resid**2)
        sigma2 = invgamma.rvs(a=alpha_post, scale=beta_post)

        # Step 4. Sample transition matrix rows
        counts = np.zeros((k_regimes, k_regimes))
        for t in range(T-1):
            counts[s_t[t], s_t[t+1]] += 1
        for s in range(k_regimes):
            P[s,:] = np.random.dirichlet(prior_trans[s,:] + counts[s,:])

        # Store samples
        alpha_samples[it,:] = alpha
        beta_samples[it,:] = beta
        sigma2_samples[it] = sigma2
        s_samples[it,:] = s_t

        if verbose and (it+1) % 100 == 0:
            print(f"Iteration {it+1} complete")

    # ---------------------------
    # 6. Post-processing
    # ---------------------------
    alpha_samples_ = alpha_samples[burn_in:]
    beta_samples_ = beta_samples[burn_in:]
    sigma2_samples_ = sigma2_samples[burn_in:]
    s_samples_ = s_samples[burn_in:]

    # Calculate posterior mean cay_MS with regime-dependent beta
    alpha_mean = alpha_samples_.mean(axis=0)
    beta_mean = beta_samples_.mean(axis=0)
    s_t_mode = np.round(s_samples_.mean(axis=0)).astype(int)
    cay_MS = y - (alpha_mean[s_t_mode] + beta_mean[s_t_mode] * X)

    # Add to dataframe
    df = df.copy()
    df[f'cay_MS_{model}_gibbs'] = cay_MS

    if verbose:
        print(f"Full Gibbs sampling with regime-dependent slopes completed. cay_MS for model {model} calculated and added to dataframe.")

    # Return results dict
    return {
        'df': df,
        'alpha_samples': alpha_samples_,
        'beta_samples': beta_samples_,
        'sigma2_samples': sigma2_samples_,
        's_samples': s_samples_
    }

def estimate_cay_MS_gibbs_macro(df, k_regimes=2, n_iter=1000, burn_in=200, model='yt', verbose=True):
    """
    Gibbs sampler for regime-switching model with macro controls as regime-invariant predictors.
    """
    # ---------------------------
    # 0. Data preparation
    # ---------------------------
    y = df['log_ct'].values
    if model == 'yt':
        X_main = df['log_yt'].values
    elif model == 'pca':
        X_main = df['pca'].values
    else:
        raise ValueError("Invalid model")

    # Macro controls
    Z = df[['interest_rate', 'CPI_inflation', 'unemployment']].values
    T = len(y)
    n_macro = Z.shape[1]

    # ---------------------------
    # 1. Hyperparameters
    # ---------------------------
    sigma2_y = np.var(y)
    mu_alpha = np.zeros(k_regimes)
    sigma2_alpha = np.ones(k_regimes) * sigma2_y
    mu_beta = np.zeros(k_regimes)
    sigma2_beta = np.ones(k_regimes) * sigma2_y

    mu_gamma = np.zeros(n_macro)
    sigma2_gamma = np.ones(n_macro) * sigma2_y

    alpha_sigma = 2.5
    beta_sigma = 0.5
    prior_trans = np.ones((k_regimes, k_regimes)) * 0.5

    # ---------------------------
    # 2. Initialize storage
    # ---------------------------
    alpha_samples = np.zeros((n_iter, k_regimes))
    beta_samples = np.zeros((n_iter, k_regimes))
    gamma_samples = np.zeros((n_iter, n_macro))
    sigma2_samples = np.zeros(n_iter)
    s_samples = np.zeros((n_iter, T))

    # Initialize parameters
    alpha = np.zeros(k_regimes)
    beta = np.zeros(k_regimes)
    gamma = np.zeros(n_macro)
    sigma2 = 1
    P = np.full((k_regimes, k_regimes), 1/k_regimes)
    s_t = np.random.choice(k_regimes, size=T)

    # ---------------------------
    # Helper functions
    # ---------------------------
    def hamilton_filter(y, X_main, Z, alpha, beta, gamma, sigma2, P, pi0):
        T = len(y)
        k = len(alpha)
        xi = np.zeros((T, k))
        for s in range(k):
            mu = alpha[s] + beta[s]*X_main[0] + np.dot(Z[0,:], gamma)
            xi[0, s] = pi0[s] * norm.pdf(y[0], loc=mu, scale=np.sqrt(sigma2))
        xi[0, :] /= xi[0, :].sum()
        for t in range(1, T):
            for s in range(k):
                mu = alpha[s] + beta[s]*X_main[t] + np.dot(Z[t,:], gamma)
                xi[t, s] = norm.pdf(y[t], loc=mu, scale=np.sqrt(sigma2)) * np.dot(xi[t-1, :], P[:, s])
            xi[t, :] /= xi[t, :].sum()
        return xi

    def backward_sampling(xi, P):
        T, k = xi.shape
        s_t = np.zeros(T, dtype=int)
        s_t[T-1] = np.random.choice(k, p=xi[T-1, :])
        for t in range(T-2, -1, -1):
            prob = xi[t, :] * P[:, s_t[t+1]]
            prob /= prob.sum()
            s_t[t] = np.random.choice(k, p=prob)
        return s_t

    # ---------------------------
    # 3. Gibbs sampler loop
    # ---------------------------
    for it in range(n_iter):
        # Step 1. Sample s_t
        pi0 = np.full(k_regimes, 1/k_regimes)
        xi = hamilton_filter(y, X_main, Z, alpha, beta, gamma, sigma2, P, pi0)
        s_t = backward_sampling(xi, P)

        # Step 2. Sample alpha_s and beta_s (regime-dependent)
        for s in range(k_regimes):
            idx = (s_t == s)
            n_s = np.sum(idx)
            if n_s > 0:
                y_s = y[idx]
                X_s = X_main[idx]
                Z_s = Z[idx,:]
                y_tilde = y_s - np.dot(Z_s, gamma)

                # Sample beta_s
                var_beta = 1 / (np.sum(X_s**2) / sigma2 + 1 / sigma2_beta[s])
                mean_beta = var_beta * (np.sum(X_s * (y_tilde - alpha[s])) / sigma2)
                beta[s] = norm.rvs(loc=mean_beta, scale=np.sqrt(var_beta))

                # Sample alpha_s
                var_alpha = 1 / (n_s / sigma2 + 1 / sigma2_alpha[s])
                mean_alpha = var_alpha * (np.sum(y_tilde - beta[s] * X_s) / sigma2)
                alpha[s] = norm.rvs(loc=mean_alpha, scale=np.sqrt(var_alpha))
            else:
                beta[s] = norm.rvs(loc=mu_beta[s], scale=np.sqrt(sigma2_beta[s]))
                alpha[s] = norm.rvs(loc=mu_alpha[s], scale=np.sqrt(sigma2_alpha[s]))

        # Enforce ordering constraint (α_0 < α_1)
        sort_idx = np.argsort(alpha)
        alpha = alpha[sort_idx]
        beta = beta[sort_idx]
        new_s_t = np.zeros_like(s_t)
        for new_label, old_label in enumerate(sort_idx):
            new_s_t[s_t == old_label] = new_label
        s_t = new_s_t

        # Step 3. Sample gamma (regime-invariant macro coefficients)
        ZTZ = np.dot(Z.T, Z)
        var_gamma = np.linalg.inv(ZTZ / sigma2 + np.diag(1 / sigma2_gamma))
        y_resid = y - (alpha[s_t] + beta[s_t] * X_main)
        mean_gamma = np.dot(var_gamma, np.dot(Z.T, y_resid) / sigma2)
        gamma = np.random.multivariate_normal(mean_gamma, var_gamma)

        # Step 4. Sample sigma2
        resid = y - (alpha[s_t] + beta[s_t]*X_main + np.dot(Z, gamma))
        alpha_post = alpha_sigma + T/2
        beta_post = beta_sigma + 0.5 * np.sum(resid**2)
        sigma2 = invgamma.rvs(a=alpha_post, scale=beta_post)

        # Step 5. Sample transition matrix rows
        counts = np.zeros((k_regimes, k_regimes))
        for t in range(T-1):
            counts[s_t[t], s_t[t+1]] += 1
        for s in range(k_regimes):
            P[s,:] = np.random.dirichlet(prior_trans[s,:] + counts[s,:])

        # Store samples
        alpha_samples[it,:] = alpha
        beta_samples[it,:] = beta
        gamma_samples[it,:] = gamma
        sigma2_samples[it] = sigma2
        s_samples[it,:] = s_t

        if verbose and (it+1) % 100 == 0:
            print(f"Iteration {it+1} complete")

    # ---------------------------
    # 4. Post-processing
    # ---------------------------
    alpha_mean = alpha_samples[burn_in:].mean(axis=0)
    beta_mean = beta_samples[burn_in:].mean(axis=0)
    gamma_mean = gamma_samples[burn_in:].mean(axis=0)

    # Reconstruct fitted values and cay_MS_macro
    fitted = alpha_mean[s_t] + beta_mean[s_t] * X_main + np.dot(Z, gamma_mean)
    cay_MS_macro = y - fitted

    # Add to dataframe
    df = df.copy()
    df[f'cay_MS_{model}_gibbs_macro'] = cay_MS_macro

    # Prepare results
    results = {
        'df': df,
        'alpha_samples': alpha_samples[burn_in:],
        'beta_samples': beta_samples[burn_in:],
        'gamma_samples': gamma_samples[burn_in:],
        'sigma2_samples': sigma2_samples[burn_in:],
        's_samples': s_samples[burn_in:]
    }

    if verbose:
        print(f"Gibbs sampling with macro controls completed for model {model}. cay_MS_macro added to dataframe.")

    return results

def estimate_cay_FC_yt(df, verbose=True):
    """
    Estimate cay_FC_yt using a simple OLS regression of log_ct on log_yt.
    Adds 'cay_FC_yt' column to df.
    """
    y = df['log_ct']
    X = df[['log_yt']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if verbose:
        print(model.summary())
    df = df.copy()
    df['cay_FC_yt'] = model.resid
    return df, model

def estimate_cay_FC_pca(df, verbose=True):
    """
    Estimate cay_FC_pca using a simple OLS regression of log_ct on the PCA factor.
    Adds 'cay_FC_pca' column to df.
    """
    y = df['log_ct']
    X = df[['pca']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    if verbose:
        print(model.summary())
    df = df.copy()
    df['cay_FC_pca'] = model.resid
    return df, model

def run_multi_chain_gibbs(
    df,
    gibbs_function,
    num_chains=4,
    num_iterations=2000,
    burn_in=200,
    k_regimes=2,
    model='yt',
    verbose=True
):
    """
    Runs multiple chains of a Gibbs sampler function and returns combined ArviZ posterior with diagnostics.

    Parameters:
    - df: input dataframe
    - gibbs_function: Gibbs sampler function to call
    - num_chains: number of independent chains
    - num_iterations: number of iterations per chain
    - burn_in: number of burn-in iterations to discard
    - k_regimes: number of regimes
    - model: model type ('yt', 'pca', etc.)
    - verbose: print iteration progress

    Returns:
    - posterior: combined ArviZ InferenceData object
    """
    
    all_chains = []
    
    for chain_id in range(num_chains):
        seed = 42 + chain_id
        np.random.seed(seed)
        
        if verbose:
            print(f"Running Chain {chain_id+1}/{num_chains} with seed {seed}")
        
        results = gibbs_function(
            df=df,
            k_regimes=k_regimes,
            n_iter=num_iterations,
            burn_in=burn_in,
            model=model,
            verbose=False
        )
        
        # Extract samples
        alpha_samples = results['alpha_samples']
        beta_samples = results['beta_samples']
        sigma2_samples = results['sigma2_samples']
        
        # Convert to dict for ArviZ
        samples_dict = {
            'alpha_0': alpha_samples[:,0],
            'alpha_1': alpha_samples[:,1],
            'beta_0': beta_samples[:,0],
            'beta_1': beta_samples[:,1],
            'sigma2': sigma2_samples
        }
        
        # Convert to InferenceData
        posterior = az.from_dict(samples_dict)
        all_chains.append(posterior)
    
    # === Combine all chains into one InferenceData object ===
    combined_posterior = az.concat(all_chains, dim="chain")
    
    # === Convergence Diagnostics ===
    
    if verbose:
        if combined_posterior is None:
            print("No posterior samples were generated.")
            return None
        # Traceplots
        az.plot_trace(combined_posterior)
        plt.tight_layout()
        plt.show()

        # Effective Sample Size (ESS)
        ess = az.ess(combined_posterior)
        print("Effective Sample Sizes (ESS):")
        print(ess)

        # R-hat diagnostic
        rhat = az.rhat(combined_posterior)
        print("R-hat Diagnostics:")
        print(rhat)

        # Summary table
        summary = az.summary(combined_posterior)
        print("Posterior Summary:")
        print(summary)
    
    return combined_posterior

def baseline_mean_forecast(df, target='future_ret_1q', window_size=40, expanding=True, save_csv=True, csv_path='results/baseline_mean_results.csv'):
    """
    Computes rolling or expanding mean forecast as a baseline model and optionally saves results.
    
    Parameters:
    - df: DataFrame with target column
    - target: str, name of the target variable (e.g. 'future_ret_1q')
    - window_size: int, size of the rolling window (ignored if expanding=True)
    - expanding: bool, use expanding window if True, rolling window if False
    - save_csv: bool, save results to CSV if True
    - csv_path: str, path to save the CSV file
    
    Returns:
    - DataFrame with actuals, forecast, errors, and performance metrics.
    """
    preds = []
    actuals = []

    for i in range(window_size, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i]
        
        if expanding:
            y_mean = train[target].mean()
        else:
            y_mean = train[target].iloc[-window_size:].mean()
        
        preds.append(y_mean)
        actuals.append(test[target])
    
    results = pd.DataFrame({
        'actual': actuals,
        'pred': preds
    })
    results['error'] = results['actual'] - results['pred']
    
    # Performance metrics
    r2 = r2_score(results['actual'], results['pred'])
    rmse = np.sqrt(np.mean(results['error'] ** 2))
    sharpe = (results['pred'].mean() / results['pred'].std()) * np.sqrt(12)  # approximate annualized

    print(f"🔷 Baseline Mean Model Performance:\nR² (OOS): {r2:.4f}\nRMSE: {rmse:.4f}\nSharpe Ratio (annualized): {sharpe:.4f}")
    
    # Save results if requested
    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results.to_csv(csv_path, index=False)
        print(f"Baseline mean forecast results saved to {csv_path}")

    return results, {'R2': r2, 'RMSE': rmse, 'Sharpe_ann': sharpe}


