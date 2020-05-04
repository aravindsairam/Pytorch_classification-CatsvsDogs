import numpy as np
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

"""
- binary cross-validate
- multi-class cross-validate
- multi-label cross-validate
- holdout
- regression
"""

class CrossValidate(object):
    def __init__(self, df,
                    target_cols,
                    problem_type,
                    num_folds = 3,
                    shuffle = False,
                    random_state = 0):
        """
        df - pandas dataframe
        target_cols - list of targets
        problem_type - ["binary", "multiclass", holdout_n, multilabel]
        """

        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.shuffle = shuffle
        self.num_folds = num_folds
        self.random_state = random_state

        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac = 1,
                                                    random_state =  self.random_state).reset_index(drop = True)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary", "multiclass"):
            """
            target_cols - ['target_1']
            unique_values - eg, [0, 1] for binary, [0, 1, 2,...] for multiclass
            """
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type. \
                                Needed number of targets = 1")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()

            if unique_values == 1:
                raise Exception("Only one unique value found! \
                                Must be two for Binary and Multiclass cross validation")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                    shuffle = False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                    y=self.dataframe[target].values)):
                        self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type == "multilabel":
            """
            target_cols - ['target_1', 'target_2', 'target_3',....]
            """
            if self.num_targets < 1:
                raise Exception("Invalid number of targets for this problem type. \
                                Must be greater than 1.")

            kf = MultilabelStratifiedKFold(n_splits=self.num_folds,
                                            shuffle = False)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                y=self.dataframe[self.target_cols].values)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("regression"):
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            """
            1 : Training Set
            0 : Validation Set
            holdout_n : n% to holdout
            """
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe
