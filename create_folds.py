import pandas as pd
from cross_validation import CrossValidate

if __name__ == "__main__":
    df = pd.read_csv("datasets/train.csv")

    kf = CrossValidate(df=df, target_cols=['label'],num_folds = 5, problem_type='binary')

    df_new = kf.split()

    df_new.to_csv("datasets/train_folds.csv", index=False)
