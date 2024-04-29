# Built-in libraries
from typing import Generator

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


# Custom libraries

class Split:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    
    def test_train(self, *args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        train_index, test_index = train_test_split(
            self.X.index.get_level_values('Index').unique(),
            *args, **kwargs
            )
        X_train = self.X.loc[train_index]
        X_test = self.X.loc[test_index]
        y_train = self.y.loc[train_index]
        y_test = self.y.loc[test_index]

        return X_train, X_test, y_train, y_test
    
    def k_fold(self, *args, **kwargs) -> Generator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
        kf = KFold(*args, **kwargs)
        for train_index, test_index in kf.split(self.X.index.get_level_values('Index').unique()):
            X_train = self.X.loc[train_index]
            X_test = self.X.loc[test_index]
            y_train = self.y.loc[train_index]
            y_test = self.y.loc[test_index]
            yield X_train, X_test, y_train, y_test

    def nested_cv(self, outer_k: int, inner_k: int, *args, **kwargs) -> Generator[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
        outer_kf = KFold(n_splits = outer_k *args, **kwargs)
        inner_kf = KFold(n_splits = inner_k, *args, **kwargs)

        for outer_train_index, outer_test_index in outer_kf.split(self.X.index.get_level_values('Index').unique()):
            X_outer_train = self.X.loc[outer_train_index]
            X_outer_test = self.X.loc[outer_test_index]
            y_outer_train = self.y.loc[outer_train_index]
            y_outer_test = self.y.loc[outer_test_index]

            for inner_train_index, inner_test_index in inner_kf.split(X_outer_train.index.get_level_values('Index').unique()):
                X_inner_train = X_outer_train.loc[inner_train_index]
                X_inner_test = X_outer_train.loc[inner_test_index]
                y_inner_train = y_outer_train.loc[inner_train_index]
                y_inner_test = y_outer_train.loc[inner_test_index]

                yield X_outer_train, X_outer_test, y_outer_train, y_outer_test, X_inner_train, X_inner_test, y_inner_train, y_inner_test, 

if __name__ == '__main__':
    iterables = [range(10), ["one", "two"]]
    index = pd.MultiIndex.from_product(iterables, names=["Index", "UniProt ID"])
    X = pd.Series(np.random.randn(20), index=index)
    y = pd.Series(np.random.randint(20), index=index)

    splitter = Split(X, y)
    X_train, X_test, y_train, y_test = splitter.test_train()
    for X_train, X_test, y_train, y_test in splitter.k_fold(n_splits=3, shuffle = True):
        print(X_train, X_test, y_train, y_test)
        print('---------------------------------------------')
