import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots



def generate_data(alpha, beta):
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

import plotly.graph_objs as go

def simulation_df(alpha, beta, sims):
    outcomes_df = []
    choices_df = []
    V_a_df = []
    V_b_df = []
    for i in range(sims):
        choice, outcomes, V_a, V_b = generate_data(alpha, beta)
        outcomes_df.append(outcomes)
        choices_df.append(choice)
        V_a_df.append(V_a)
        V_b_df.append(V_b)
    df_sims = pd.DataFrame(outcomes_df, columns=[str(i) for i in range(1, 160 + 1)])
    average_ones_per_row = (df_sims == 1).mean(axis=1)
    sum_ones = (df_sims == 1).sum(axis=1)
    df_sims['average_ones_per_row'] = average_ones_per_row
    df_sims['sum_ones_per_row'] = sum_ones

    # Create a dataframe for Va and Vb
    df_va = pd.DataFrame(V_a_df, columns=[str(i) for i in range(1, 160 + 1)])
    df_vb = pd.DataFrame(V_b_df, columns=[str(i) for i in range(1, 160 + 1)])

    # Create Va minus Vb
    df_a_minus_b = df_va - df_vb

    # Calculate average evolution for Va, Vb, and Va minus Vb
    average_evolution_va = df_va.mean(axis=0)
    average_evolution_vb = df_vb.mean(axis=0)
    average_evolution_a_minus_b = df_a_minus_b.mean(axis=0)

    # Add average evolution values to the dataframes
    df_va.loc['mean'] = average_evolution_va
    df_vb.loc['mean'] = average_evolution_vb
    df_a_minus_b.loc['mean'] = average_evolution_a_minus_b

    # Plot the series over time using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_va.columns, y=df_va.loc['mean'], name='average Va'))
    fig.add_trace(go.Scatter(x=df_vb.columns, y=df_vb.loc['mean'], name='average Vb'))
    fig.add_trace(go.Scatter(x=df_a_minus_b.columns, y=df_a_minus_b.loc['mean'], name='Va-Vb'))

    # Customize the plot
    fig.update_layout(title='Strength of Stimuli over trials', xaxis_title='Trials', yaxis_title='Value', legend=dict(x=1, y=0))

    # Show the plot
    fig.show()

    return df_sims, df_va, df_vb, df_a_minus_b





    

def observation(V_init,alpha,beta,t,outcome):
    V = np.zeros(t)
    V[0] = V_init
    for i in range(1,t):
        V[t] = V[t-1] + alpha * (outcome[t-1]-V[t-1])





def create_outcomes_df():
    # Initialize an empty list to store the outcomes
    all_outcomes = []
    
    # Generate 50 different outcomes using the generate_outcomes() function
    for i in range(50):
        outcomes = generate_outcomes()
        all_outcomes.append(outcomes)
    
    # Create a Pandas DataFrame from the outcomes
    df = pd.DataFrame(all_outcomes)
    
    return df
