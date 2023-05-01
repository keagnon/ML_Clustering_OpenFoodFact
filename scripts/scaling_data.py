import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

class Scaling:
    """Class to scaling pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
        method (string): method to scale pandas dataframe
    Returns:
        df (DataFrame): return scaled pandas dataframe
    """

    def __init__(
                    self,
                    df     = None,
                    method = 'standard'
                ):

        self.df     = df
        self.method = method

    def convert_numpy_to_pandas(self, np_array):
        """Convert numpy array to pandas Dataframe
        Args:
            np_array (array) : numpy array
        Returns:
            df (DataFrame): return pandas dataframe
        @Author: Thomas PAYAN
        """
        return pd.DataFrame(np_array)

    def convert_categorical_features_to_numeric(self):
        """Convert categorical features to numeric
        Returns:
            df (DataFrame): return dataframe with categorical features converted
        """
        print("\nPerforming categorical features convertion")

        df_cat = self.df.select_dtypes(include=["object"])

        for col in df_cat.columns.tolist():
            self.df[col] = self.df[col].astype('category')
            self.df[col] = self.df[col].cat.codes

        return self.df

    def standard_scaler(self, **kwargs):
        """Scale dataframe features with StandardScaler transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nStandard scaling")
        with_mean = kwargs.get('with_mean', True)
        with_std  = kwargs.get('with_std', True)
        scaled_df = StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def min_max_scaler(self, **kwargs):
        """Scale dataframe features with MinMaxScaler transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nMin-max scaling")
        feature_range = kwargs.get('feature_range', (0,1))
        scaled_df     = MinMaxScaler(feature_range=feature_range).fit_transform(self.df)
        scaled_df     = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def max_abs_scaler(self, **kwargs):
        """Scale dataframe features with MaxAbsScaler transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nMax-abs scaling")
        copy      = kwargs.get('copy', True)
        scaled_df = MaxAbsScaler(copy=copy).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def robust_scaler(self, **kwargs):
        """Scale dataframe features with RobustScaler transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nRobust scaling")
        with_centering = kwargs.get('with_centering', True)
        with_scaling   = kwargs.get('with_scaling', True)
        quantile_range = kwargs.get('quantile_range', (25.0, 75.0))
        scaled_df      = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range).fit_transform(self.df)
        scaled_df      = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def power_transformation(self, **kwargs):
        """Scale dataframe features with Power transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nPower transformation ("+method+")")
        method      = kwargs.get('method', 'yeo-johnson')
        standardize = kwargs.get('standardize', True)
        scaled_df   = PowerTransformer(method=method, standardize=standardize).fit_transform(self.df)
        scaled_df   = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def quantile_transformation(self, **kwargs):
        """Scale dataframe features with Quantile transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nQuantile transformation ("+output_distribution+")")
        n_quantiles         = kwargs.get('n_quantiles', 200)
        output_distribution = kwargs.get('output_distribution', 'yeo-johnson')
        scaled_df           = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution).fit_transform(self.df)
        scaled_df           = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df
    
    def normalize_transformation(self, **kwargs):
        """Scale dataframe features with Normalizer transformation
        Args:
            kwargs (any): method parameters
        Returns:
            scaled_df (DataFrame): return new scaled dataframe
        """
        print("\nNormalize transformation")
        norm      = kwargs.get('norm', "l2")
        scaled_df = Normalizer(norm=norm).fit_transform(self.df)
        scaled_df = self.convert_numpy_to_pandas(scaled_df)
        return scaled_df

    def scaling_features(self, **kwargs):
        """Scale dataframe features
        Args:
            kwargs (any): method parameters
        Returns:
            df (DataFrame): return scaled dataframe features
        """
        print("\nPerforming features scaling")

        match self.method:
            case 'standard':
                self.df = self.standard_scaler(**kwargs)
            case 'min_max':
                self.df = self.min_max_scaler(**kwargs)
            case 'max_abs':
                self.df = self.max_abs_scaler(**kwargs)
            case 'robust':
                self.df = self.robust_scaler(**kwargs)
            case 'power':
                self.df = self.power_transformation(**kwargs)
            case 'quantile':
                self.df = self.quantile_transformation(**kwargs)
            case 'normalize':
                self.df = self.normalize_transformation(**kwargs)
            case _:
                print("\nWarning : select another method !")

    def scaling(self):
        """Scale dataframe
        Returns:
            df (DataFrame): return scaled dataframe
        """
        self.convert_categorical_features_to_numeric()

        self.scaling_features()

        return self.df
