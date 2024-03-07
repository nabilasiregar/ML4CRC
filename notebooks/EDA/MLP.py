import numpy as np
from scipy.stats import norm, poisson, expon
from scipy.optimize import minimize
import pandas as pd


# Generate example data
file_path = r'/home/rodrigo/Documents/Bioinformatics_&_SB/S-ML/ML4CRC/data/reduced_tcga_rna_count_data_crc.csv'
data = pd.read_csv(file_path)
data = data.iloc[1:, 1:]

# Define negative log-likelihood functions for each distribution
def neg_log_likelihood_gaussian(params, data):
    mu, sigma = params
    log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    return -log_likelihood

def neg_log_likelihood_poisson(params, data):
    lam = params
    log_likelihood = np.sum(poisson.logpmf(data, mu=lam))
    return -log_likelihood

def neg_log_likelihood_exponential(params, data):
    lam = params
    log_likelihood = np.sum(expon.logpdf(data, scale=1/lam))
    return -log_likelihood

# Initial guess for parameters
initial_params = [0, 1]

# Fit Gaussian distribution
result_gaussian = minimize(neg_log_likelihood_gaussian, initial_params, args=(data,))
mu_gaussian, sigma_gaussian = result_gaussian.x
likelihood_gaussian = -result_gaussian.fun

# Fit Poisson distribution
result_poisson = minimize(neg_log_likelihood_poisson, [1], args=(data,))
lam_poisson = result_poisson.x[0]
likelihood_poisson = -result_poisson.fun

# Fit Exponential distribution
result_exponential = minimize(neg_log_likelihood_exponential, [1], args=(data,))
lam_exponential = result_exponential.x[0]
likelihood_exponential = -result_exponential.fun

# Print estimated parameters and likelihoods
print("Gaussian Distribution:")
print("Estimated mu:", mu_gaussian)
print("Estimated sigma:", sigma_gaussian)
print("Log-Likelihood:", likelihood_gaussian)

print("\nPoisson Distribution:")
print("Estimated lambda:", lam_poisson)
print("Log-Likelihood:", likelihood_poisson)

print("\nExponential Distribution:")
print("Estimated lambda:", lam_exponential)
print("Log-Likelihood:", likelihood_exponential)