import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


drop_col = [ "Topic", "Class", "Stratification2", "Stratification1", "Topic", "LocationDesc"]
df = pd.read_csv('data.csv', usecols=lambda x: x in drop_col)  
df = df.replace(',','', regex=True)
# df.head()
df = df.dropna()
# df = df.sample(frac=.1)
df["transaction"] = df["Topic"].astype(str) +","+ df["Stratification1"].astype(str)+","+ df["Stratification2"].astype(str)+","+ df["LocationDesc"].astype(str)
# print(df.head())
data = list(df["transaction"].apply(lambda x:x.split(",") ))
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df = apriori(df, min_support = 0.01, use_colnames = True, verbose = 1)
df.to_csv("association_support_.csv")
# print(df)
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.1)
df_ar.to_csv("association_rules_.csv")
# print(df_ar)