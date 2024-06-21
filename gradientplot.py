import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Gradient dataframe.csv")
hist = pd.df.hist(columns=[0])
ax = df.plot.hist(bins=100, alpha=0.5)