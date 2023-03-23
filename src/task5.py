from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.task2 import *

# create a multi-variate dist of alpha and  beta values using the means from the data

def multi_variate(mean_alpha,mean_beta,var_a,var_b):
    mu = np.array([mean_alpha, mean_beta])
    sigma = np.array([[var_a,0], [0,var_b]])

    # Generate a multivariate normal distribution with 1000 samples
    samples = np.random.multivariate_normal(mu, sigma, size=50)
    alpha_values = samples[:, 0]
    beta_values = samples[:,1]
    

        # Plot alphas
    plt.scatter(np.arange(len(alpha_values)), alpha_values, label='Alpha')
    plt.xlabel('Participant index')
    plt.ylabel('Alpha value')
    plt.legend()
    plt.show()

    # Plot betas
    plt.scatter(np.arange(len(beta_values)), beta_values, label='Beta')
    plt.xlabel('Participant index')
    plt.ylabel('Beta value')
    plt.legend()
    plt.show()

    return samples

def param_reconstruction(samples):
    choices=[]
    outcomes=[]
    for i in samples:
        choice,outcome,V_a,V_b=generate_data(i[0], i[1])
        choices.append(choice)
        outcomes.append(outcome)
    choice_simulations=pd.DataFrame(choices)
    outcomes_simulations=pd.DataFrame(outcomes)

    return choice_simulations,outcomes_simulations

def og_recon_pearson(sample,reconstructed):
    corr, p_value = pearsonr(alpha_values, beta_values)
