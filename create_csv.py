import pandas as pd
import os

#dog = 1, cat = 0

#wrong or mislabelled images
wrong_df = pd.read_csv('datasets/wrong_label.csv')


wrong_df['filename'] = wrong_df['filename']+'.jpg'

wrong_fn = wrong_df['filename'].to_list()


folder = os.listdir('datasets/train')

new_files = [x for x in folder if x not in wrong_fn]

df = pd.DataFrame({"files":new_files})

df['label'] = -1

df.loc[df['files'].str.replace(r"dog.\d*.jpg+", "1", regex=True)=="1", "label"] = 1

df.loc[df['files'].str.replace(r"cat.\d*.jpg+", "0", regex=True)=="0", "label"] = 0


df.to_csv("datasets/train.csv", index = False)