from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd

class MyPandasQuantileTransformer(BaseEstimator, TransformerMixin):
    """Transform all numeric, non-categorical variables according to 
    QuantileTransformer. Return pandas DataFrame"""
    def __init__(self, n_quantiles=100, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_dist = output_distribution

    def fit(self, X, y=None):
        self.transformer = QuantileTransformer(n_quantiles = self.n_quantiles,
                                               output_distribution = self.output_dist)
        self.cont_cols = get_cont_cols(X)
        self.transformer.fit(X[self.cont_cols])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.cont_cols] = self.transformer.transform(X[self.cont_cols])
        return X

class MyPandasPCA(BaseEstimator, TransformerMixin):
    """Transform `X` according to PCA. Return pandas DataFrame"""
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.input_columns = np.array(X.columns)
        self.transformer = PCA(n_components=self.n_components, svd_solver='full')
        self.transformer.fit(X[self.input_columns], y)
        self.n_cols = len(self.transformer.components_)
        self.output_columns = [f'pca{ii+1:02}' for ii in range(self.n_cols)]
        return self


    def transform(self, X, y=None):
        X_trans = self.transformer.transform(X[self.input_columns])
        n_cols = X_trans.shape[1]
        df_out = pd.DataFrame(X_trans, columns=self.output_columns)
        return df_out


class MyPandasVarianceThreshold(BaseEstimator, TransformerMixin):
    """Select columns with variance above `threshold`. Return pandas DataFrame"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.transformer = VarianceThreshold(threshold=self.threshold)
        self.transformer.fit(X, y)
        self.input_columns = np.array(X.columns)
        support = self.transformer.get_support()
        self.output_columns = self.input_columns[support]
        return self

    def transform(self, X, y=None):
        X_trans = self.transformer.transform(X)
        df_out = pd.DataFrame(X_trans, columns=self.output_columns)
        return df_out



class MyPandasPolynomialFeatures(BaseEstimator, TransformerMixin):
    """Make polynomial interactions according to PolynomialFeatures.
    Returns pandas DataFrame"""
    def __init__(self, interaction_only=True, include_bias=False):
        self.interaction_only=interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.transformer = PolynomialFeatures(interaction_only=self.interaction_only,
                                              include_bias=self.include_bias)
        self._transformer = clone(self.transformer)
        self.input_columns = list(X.columns)
        self._transformer.fit(X)
        return self

    def transform(self, X, y=None):
        transformed_col_names = self._transformer.get_feature_names(self.input_columns)
        X_trans = self._transformer.transform(X)
        df_out = pd.DataFrame(X_trans, columns=transformed_col_names)
        return df_out



class MyPandasStandardScaler(BaseEstimator, TransformerMixin):
    """Transform all numeric, non-categorical variables according to 
    StandardScaler. Return pandas DataFrame"""

    def __init__(self, **kwargs):
        self.scaler = StandardScaler(**kwargs)
        
    def fit(self, X, y=None):
        self.cont_cols = get_cont_cols(X)
        self.scaler.fit(X[self.cont_cols])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.cont_cols] = self.scaler.transform(X[self.cont_cols])
        return X


class SingleColInteractions(BaseEstimator, TransformerMixin):
    """Make interaction terms between column `col_constant`, and all the columns 
    in `col_interactions`. Returns pandas DataFrame 
    """
    def __init__(self, col_constant, cols_interactions):
        self.col_constant = col_constant
        self.cols_interactions = cols_interactions
        assert isinstance(cols_interactions, list)

        self.interaction_terms = []
        for col in self.cols_interactions:
            interaction_name = f'{self.col_constant} x {col}'
            self.interaction_terms.append(interaction_name)
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        assert isinstance(X, pd.DataFrame)
        for col in self.cols_interactions:
            interaction_name = f'{self.col_constant} x {col}'
            X[interaction_name] = X[self.col_constant] * X[col]
        return X


    def get_feature_names(self):
        return self.interaction_terms


def get_cont_cols(df):
    """Return column names in `df` for numeric, continious columns,
    with no missing values"""
    cont_cols = []
    for col in df:
        if len(df[col].unique()) > 2 and df[col].isnull().sum() == 0 and is_numeric_dtype(df[col]):
            cont_cols.append(col)
    return cont_cols