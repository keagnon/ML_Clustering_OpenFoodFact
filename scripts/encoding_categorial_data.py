import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class encoding:
    """Class to preprocessing pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
    Returns:
        df (DataFrame): return preprocessed pandas dataframe
    """

    def __init__(
                    self,
                    df                  = None,
                    label_encode_method = 'one_hot'
                ):

        self.df                  = df
        self.label_encode_method = label_encode_method

    def convert_numpy_to_pandas(self, np_array):
        """Convert numpy array to pandas Dataframe
        Args:
            np_array (array) : numpy array
        Returns:

        """
        return pd.DataFrame(np_array)
    
    def code_encoding(self):
        """Encode categorical features using category codes
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        """
        print("\nCategory codes encoding")

        df_cat = self.df.select_dtypes(include=["object"])

        for col in df_cat.columns.tolist():
            self.df[col] = self.df[col].astype('category')
            self.df[col] = self.df[col].cat.codes

        return self.df
    
    def label_encoding(self):
        """Encode categorical features using LabelEncoder
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        """
        print("\nLabel encoding")

        df_cat = self.df.select_dtypes(include=["object"])

        for cat in df_cat.columns.tolist():
            self.df[cat] = LabelEncoder().fit_transform(self.df[cat])

        return self.df
    
    def one_hot_encoding(self, **kwargs):
        """Encode categorical features using OneHotEncoder
        Args:
            kwargs (any): method parameters
        Returns:
            df (DataFrame): return dataframe with categorical features encoded
        """
        print("\nOneHot encoding")

        handle_unknown = kwargs.get('handle_unknown', 'ignore')
        categories     = kwargs.get('categories', 'auto')
        sparse_output  = kwargs.get('sparse_output', True)
        encoded_df     = OneHotEncoder(handle_unknown=handle_unknown, categories=categories, sparse_output=sparse_output).fit_transform(self.df).toarray()
        self.df        = self.convert_numpy_to_pandas(encoded_df)
        return self.df
    
    def categorical_features_encoding(self, **kwargs):
        """Categorical features encoding
        Args:
            kwargs (any): method parameters
        Returns:
            df (DataFrame): return new encoding dataframe
        """
        print("\nPerforming categorical features encoding")

        match self.label_encode_method:
            case 'code':
                self.code_encoding()
            case 'label':
                self.label_encoding()
            case 'one_hot':
                self.one_hot_encoding(**kwargs)
            case _:
                print("\nWarning : select another method !")

        return self.df
    
        
    def preprocessing(self):
        """Preprocess dataframe
        Returns:
            df (DataFrame): return preprocessed dataframe
        """

        self.categorical_features_encoding()
    
        return self.df
        
