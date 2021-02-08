import numpy as np
import pandas as pd


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


file = '../data/fungi_data.csv'
df = pd.read_csv(file)
df['width-n'] = softmax(df['width'])
df['tolerance'] = df['ranking'] - df['width-n']
