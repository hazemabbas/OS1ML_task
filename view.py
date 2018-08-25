import pandas as pd
import matplotlib as plt

fname = 'data.csv'
data = pd.read_csv(fname)
data.head()
data.count()