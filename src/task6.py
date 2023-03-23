import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


def NLL_a(choices,outcomes,params):
    choice_map = {1:0,2:1}
    # need the outcomes to generate the values for v. these are used
    #to determine the probabilites  of getting a choice
    nll=0

    v = np.zeros(2) + params[2] 
    for outcome,choice in enumerate(choices):
        
        choice = choice_map[choice]
       
        other_choice = 1-choice
        prob_A = np.exp(-params[1] * v[0]) / (np.exp(-params[1] * v[0])+np.exp(-params[1]*v[1]))
        prob_B = 1-prob_A
        
        v[choice] += params[3]*params[0]*(outcomes[outcome]-v[choice])
        if choice==0:
           
            nll+=np.log(prob_A)
        else:
            
            nll+=np.log(prob_B)

    

    return -nll

def whole_df_alt(choices_df,outcomes_df,stai):
    params=[0.4,7,0.5,0.5]
    nll=[]
    for index, row in choices_df.iterrows():
        nll.append(NLL_a(row,outcomes_df.iloc[index],params))
    stai['Negative Log Likelihood']=nll
    return stai


def ideal_params_alt(choices,outcomes):
    # Define the initial guess for the parameters
    params0 = [0.4, 7, 0.5,0.5]

    def objective(params):
        return NLL_a(choices,outcomes,params)


    # Use the Nelder-Mead optimization function to minimize the negative log-likelihood
    res = minimize(objective, params0, method='Nelder-Mead', options={'disp': True})

    print('Optimized parameters:', res.x)
    return res

def param_finder_alt(choices_df, outcomes_df):
    optimized_alpha = []
    optimized_beta = []
    optimized_vo = []
    optimized_A = []
    nll=[]
    
    for index, row in choices_df.iterrows():

        res = ideal_params_alt(row.values, outcomes_df.iloc[index])
        optimized_alpha.append(res.x[0])
        optimized_beta.append(res.x[1])
        optimized_vo.append(res.x[2])
        optimized_A.append(res.x[3])
        nll.append(res.fun)

    df = pd.DataFrame({
        'optimized_alpha': optimized_alpha,
        'optimized_beta': optimized_beta,
        'optimized_vo': optimized_vo,
        'optimized_A': optimized_A,
        'nll':nll
    })
    return df

def model_fitting_alt(df):


    # Assume that param_finder() returns a DataFrame called df

    # Extract the fitted parameter values from the DataFrame
    alpha_values = df['optimized_alpha'].values
    beta_values = df['optimized_beta'].values
    A_values = df['optimized_A'].values

    # Calculate mean and variance of the fitted parameter values
    alpha_mean = np.mean(alpha_values)
    alpha_var = np.var(alpha_values)
    beta_mean = np.mean(beta_values)
    beta_var = np.var(beta_values)
    A_mean = np.mean(A_values)
    A_var = np.mean(A_values)


    print(f"Mean alpha value: {alpha_mean:.2f}")
    print(f"Variance of alpha values: {alpha_var:.2f}")
    print(f"Mean beta value: {beta_mean:.2f}")
    print(f"Variance of beta values: {beta_var:.2f}")
    print(f"Mean A value: {A_mean:.2f}")
    print(f"Variance of A values: {A_var:.2f}")
    

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

      # Plot Carry over param
    plt.scatter(participant_index, A_values, label='A')
    plt.xlabel('Participant index')
    plt.ylabel('A')
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



def model_comparison(df1,df2):
    # Merge the data frames by participant ID
   

    # Compute the mean negative log-likelihood for each model
    mean_nll_model1 = df1['nll'].mean()
    mean_nll_model2 = df2['nll'].mean()

    # Compute the AIC and BIC scores for each participant and each model
    df1['aic_model1'] = 2 * df1['nll'] + 2 * 3
    df1['bic_model1'] = 2 * df1['nll'] + 3 * np.log(160)

    df2['aic_model2'] = 2 * df2['nll'] + 2 * 4
    df2['bic_model2'] = 2 * df2['nll'] + 4 * np.log(160)

    # Sum up the AIC and BIC scores for each model
    sum_aic_model1 = df1['aic_model1'].sum()
    sum_bic_model1 = df1['bic_model1'].sum()

    sum_aic_model2 = df2['aic_model2'].sum()
    sum_bic_model2 = df2['bic_model2'].sum()

    # Print the results
    print('Mean negative log-likelihood (model 1):', mean_nll_model1)
    print('Mean negative log-likelihood (model 2):', mean_nll_model2)

    print('Sum of AIC scores (model 1):', sum_aic_model1)
    print('Sum of BIC scores (model 1):', sum_bic_model1)

    print('Sum of AIC scores (model 2):', sum_aic_model2)
    print('Sum of BIC scores (model 2):', sum_bic_model2)








































def generate_data_alternative(alpha, beta):
    # Define initial values
    # 0 is A and 1 is B
    v = np.zeros(2) + 0.5  # initial values of V for each stimulus
    probs_list = [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4), (0.65, 0.35)]
    outcomes = []
    choice = []
    V_a = []
    V_b = []
    
    # Loop through trials and generate outcomes
    for i in range(160):
        
        
        # Determine which stimulus to choose based on probability and value
        prob_A = np.exp(-beta * v[0]) / (np.exp(-beta * v[0])+np.exp(-beta*v[1]))
        prob_B = 1-prob_A
        
        
        action_probs = np.exp(-beta * v) / np.sum(np.exp(-beta * v))
        
        chosen_action = np.random.choice([0, 1], p=[prob_A,prob_B])
        
        choice_action_changed = 1 if chosen_action==0 else 2
        choice.append(choice_action_changed)
        
        other_action = 1 - chosen_action
        
        
        # Get the probabilities for the current trial
        trial_index = i // 40
        probs = probs_list[trial_index]
       
        # Determine outcome based on chosen stimulus
        outcome = np.random.choice([1,0 ], p=[probs[chosen_action], probs[other_action]])
        
        # Update value of chosen stimulus based on outcome and learning rate
        v[chosen_action] += alpha * (outcome - v[chosen_action])

        V_a.append(v[0])
        V_b.append(v[1])

        
        # Append outcome to outcomes list
        outcomes.append(outcome)
    
        
    return choice,outcomes,V_a,V_b