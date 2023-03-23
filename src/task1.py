import numpy as np
import pandas as pd

def load(df):

    stai = pd.read_csv(df, header=0)
    stai.columns = ['score']
    return stai

def filter_healthy(stai):

    stai_filtered = stai[stai['score'] < 43]
    high_anxious = stai[:25]
    low_anxious = stai[-25:]

    return stai_filtered,high_anxious,low_anxious

def number_choosing_a(filename):

    choices = pd.read_csv(filename,header=None)
    choices.columns = choices.columns = [str(i) for i in range(1, len(choices.columns) + 1)]
    average_ones_per_row = (choices == 1).mean(axis=1)
    sum_of_one = (choices == 1).sum(axis=1)
    choices['average_ones_per_row'] = average_ones_per_row
    choices['sum_of_one']= sum_of_one
    seventy_thirty = choices.iloc[:,:40]
    eighty_twenty = choices.iloc[:,40:81]
    sixty_forty = choices.iloc[:,81:121]
    sixtyfive_thirtyfive = choices.iloc[:,121:160]

    return choices,seventy_thirty,eighty_twenty,sixty_forty,sixtyfive_thirtyfive






