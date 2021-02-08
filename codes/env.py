import pandas as pd
import numpy as np
import markov


def renv(r, d, w: int):
    ls = []
    for i in range(len(r)):
        ls.append(r[i] * d[i] / (w + d[i]))
    return ls


