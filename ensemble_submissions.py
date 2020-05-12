import numpy as np
import pandas as pd
from functools import reduce

df0 = pd.read_csv("submissions/submission__val_0.csv")
df1 = pd.read_csv("submissions/submission__val_1.csv")
df2 = pd.read_csv("submissions/submission__val_2.csv")
df3 = pd.read_csv("submissions/submission__val_3.csv")
df4 = pd.read_csv("submissions/submission__val_4.csv")

dfs = [df0, df1, df2, df3, df4]
df_final = reduce(lambda left,right: pd.merge(left,right,on='id'), dfs)

df_final['label'] = df_final[['label_0', 'label_1','label_2','label_3','label_4']].mean(axis=1)

mean_endf = df_final[['id','label']]

mean_endf.to_csv("submissions/mean_ensemble_submission.csv", index=False)
