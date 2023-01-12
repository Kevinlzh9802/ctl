import pandas as pd
import numpy as np
df = pd.read_csv('/Users/chenyuzhao/Downloads/weatherAUS.csv')

res = list(df['MaxTemp'])[33:3041]

for i in range(9):
    print(i, np.mean(res[365*i:365*i+365]))