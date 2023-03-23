import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

def outcomes_df(file):
    df = pd.read_csv(file,header=None)
    df.columns = [str(i) for i in range(1, len(df.columns) + 1)]

    return df

def NLL(choices,outcomes,params):
    choice_map = {1:0,2:1}
    # need the outcomes to generate the values for v. these are used
    #to determine the probabilites  of getting a choice
    nll=0
    V_a = []
    V_b = []
    v = np.zeros(2) + params[2] 
    for outcome,choice in enumerate(choices):
        
        choice = choice_map[choice]
       
        other_choice = 1-choice
        prob_A = np.exp(-params[1] * v[0]) / (np.exp(-params[1] * v[0])+np.exp(-params[1]*v[1]))
        prob_B = 1-prob_A
        
        v[choice] += params[0]*(outcomes[outcome]-v[choice])
        if choice==0:
           
            nll+=np.log(prob_A)
        else:
            
            nll+=np.log(prob_B)

    

    return -nll

def whole_df(choices_df,outcomes_df,stai):
    params=[0.4,7,0.5]
    nll=[]
    for index, row in choices_df.iterrows():
        nll.append(NLL(row,outcomes_df.iloc[index],params))
    stai['Negative Log Likelihood']=nll
    return stai

      






def ideal_params(choices,outcomes):
    # Define the initial guess for the parameters
    params0 = [0.4, 7, 0.5]

    def objective(params):
        return NLL(choices,outcomes,params)


    # Use the Nelder-Mead optimization function to minimize the negative log-likelihood
    res = minimize(objective, params0, method='Nelder-Mead', options={'disp': True})

    print('Optimized parameters:', res.x)
    return res

def param_finder(choices_df, outcomes_df):
    optimized_alpha = []
    optimized_beta = []
    optimized_vo = []
    nll=[]
    
    for index, row in choices_df.iterrows():

        res = ideal_params(row.values, outcomes_df.iloc[index])
        nll.append(res.fun)
        optimized_alpha.append(res.x[0])
        optimized_beta.append(res.x[1])
        optimized_vo.append(res.x[2])

    df = pd.DataFrame({
        'optimized_alpha': optimized_alpha,
        'optimized_beta': optimized_beta,
        'optimized_vo': optimized_vo,
        'nll':nll
    })
    return df

def model_fitting(df):


    # Assume that param_finder() returns a DataFrame called df

    # Extract the fitted parameter values from the DataFrame
    alpha_values = df['optimized_alpha'].values
    beta_values = df['optimized_beta'].values

    # Calculate mean and variance of the fitted parameter values
    alpha_mean = np.mean(alpha_values)
    alpha_var = np.var(alpha_values)
    beta_mean = np.mean(beta_values)
    beta_var = np.var(beta_values)

    print(f"Mean alpha value: {alpha_mean:.2f}")
    print(f"Variance of alpha values: {alpha_var:.2f}")
    print(f"Mean beta value: {beta_mean:.2f}")
    print(f"Variance of beta values: {beta_var:.2f}")

    # Create a scatter plot of parameter values
    participant_index = np.arange(len(alpha_values))

    # Plot alphas
    plt.scatter(participant_index, alpha_values, label='Alpha')
    plt.xlabel('Participant index')
    plt.ylabel('Alpha value')
    plt.legend()
    plt.show()

    # Plot betas
    plt.scatter(participant_index, beta_values, label='Beta')
    plt.xlabel('Participant index')
    plt.ylabel('Beta value')
    plt.legend()
    plt.show()

    # Calculate Pearson's correlation coefficient between alpha and beta across all participants
    corr, p_value = pearsonr(alpha_values, beta_values)
    print(f"Pearson's correlation coefficient between alpha and beta: {corr:.2f}")

    # Define high anxious group as first 25 participants
    high_anxious_alpha = alpha_values[:25]
    high_anxious_beta = beta_values[:25]

    # Calculate Pearson's correlation coefficient between alpha and beta within high anxious group
    corr_high_anxious, p_value_high_anxious = pearsonr(high_anxious_alpha, high_anxious_beta)
    print(f"Pearson's correlation coefficient between alpha and beta within high anxious group: {corr_high_anxious:.2f}")

    # Calculate Pearson's correlation coefficient between alpha and beta within low anxious group
    low_anxious_alpha = alpha_values[25:]
    low_anxious_beta = beta_values[25:]
    corr_low_anxious, p_value_low_anxious = pearsonr(low_anxious_alpha, low_anxious_beta)
    print(f"Pearson's correlation coefficient between alpha and beta within low anxious group: {corr_low_anxious:.2f}")


    # t-test between two groups
    alpha_tstat, alpha_pval = ttest_ind(high_anxious_alpha, low_anxious_alpha, equal_var=False)
    beta_tstat, beta_pval = ttest_ind(high_anxious_beta, low_anxious_beta, equal_var=False)
    print("Alpha t-statistic:", alpha_tstat)
    print("Alpha degrees of freedom:", len(high_anxious_alpha) + len(low_anxious_alpha) - 2)
    print("Alpha p-value:", alpha_pval)

    print("Beta t-statistic:", beta_tstat)
    print("Beta degrees of freedom:", len(high_anxious_beta) + len(low_anxious_beta) - 2)
    print("Beta p-value:", beta_pval)

    """If the p-value is less than the significance level (usually 0.05), we reject the null hypothesis and conclude that there is a significant difference in the means of the two groups. If the p-value is greater than the significance level, we fail to reject the null hypothesis and conclude that there is insufficient evidence to suggest that the means of the two groups are different.

In this case, if we obtain a significant p-value, it would mean that there is a significant difference in the alpha and/or beta values between high anxious and low anxious participants. This would support our hypothesis that anxiety levels are associated with differences in cognitive control mechanisms.

"""

            





        
