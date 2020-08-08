import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transformation of the numerical variables"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.log1p(X[feature])
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Impute the missing numerical variables with
    the group mode of type"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X.groupby('type')[feature].transform(lambda x: x.fillna(x.mode()[0]))
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.05, variables=None):
        self.encoder_dict_ = {}
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                                                      feature]), X[feature], 'Rare')
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.encoder_dict_ = {}
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']
        # persist transforming dictionary
        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]
