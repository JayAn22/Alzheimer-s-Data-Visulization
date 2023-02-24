import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

drop_col = [ "Data_Value", "Low_Confidence_Limit", "High_Confidence_Limit"]
df = pd.read_csv('data.csv', usecols=lambda x: x in drop_col) 
# df = df.sample(frac=.3)
df = df.replace('.',np.nan, regex=True)
df = df.dropna()
print(df)

y = df["Data_Value"]
x = df["Low_Confidence_Limit"]

area = (50)  # 0 to 15 point radii

plt.scatter(x, y, s=area, alpha=0.5)
plt.ylabel('Data Value',size=16)
# set y-axis label and specific size
plt.xlabel('Low Confidence Limit',size=16)
plt.show()




