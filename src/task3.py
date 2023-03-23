import plotly.graph_objs as go
from src.task2 import *
from src.task1 import *

def plot_alpha_beta():
    # Define ranges for alpha and beta
    alpha_range = np.linspace(0, 1, 10)
    beta_range = np.linspace(0, 10, 10)
    
    # Create empty arrays to store data
    avg_stimuli = np.zeros((len(alpha_range), len(beta_range)))
    
    # Loop through parameter settings and simulate data
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Simulate data for current parameter settings
            create_outcomes_df = []
            for sim in range(1,100):

                _, outcomes, _, _ = generate_data(alpha, beta)
                #df_sims = pd.DataFrame(outcomes, columns=[str(i) for i in range(1, 160 + 1)])
                ones=outcomes.count(1)
                create_outcomes_df.append(ones)

                #sum_ones = (df_sims == 1).sum(axis=1)
                #df_sims['sum_ones_per_row'] = sum_ones

            
            # Calculate average number of aversive stimuli
            avg_stimuli[i, j] = np.mean(create_outcomes_df)
    
    # Create 3D surface plot of average number of aversive stimuli
    fig = go.Figure(data=[go.Surface(x=beta_range, y=alpha_range, z=avg_stimuli)])
    
    # Set axis labels
    fig.update_layout(scene=dict(xaxis_title='Beta', yaxis_title='Alpha', zaxis_title='Avg. Aversive Stimuli'))
    
    # Show plot
    fig.show()
