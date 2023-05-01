import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

class Preprocessing:
    """Class to preprocessing pandas dataframe
    Args:
        df (DataFrame): pandas dataframe
        percent (integer): percent of missing data
        num_imput (string): method to impute numerical features missing values
        cat_imput (string): method to impute categorical features missing values
    Returns:
        df (DataFrame): return preprocessed pandas dataframe
    """

    def __init__(
                    self,
                    df                  = None,
                    percent             = 70,
                    num_imput           = 'mean',
                    cat_imput           = 'mode',
                    label_encode_method = 'one_hot'
                ):

        self.df                  = df
        self.percent             = percent
        self.num_imput           = num_imput
        self.cat_imput           = cat_imput
        self.label_encode_method = label_encode_method

    def convert_numpy_to_pandas(self, np_array):
        """Convert numpy array to pandas Dataframe
        Args:
            np_array (array) : numpy array
        Returns:
            df (DataFrame): return pandas dataframe
        """
        return pd.DataFrame(np_array)
        
    def count_duplicated_values(self):
        """Count duplicated values for each feature
        Returns:
            df (DataFrame): return sum of duplicated values
        """
        return self.df.duplicated().sum()

    def drop_duplicated_values(self):
        """Drop duplicated values
        Returns:
            df (DataFrame): return dataframe without duplicated values
        """
        print("\nRows number before drop duplicated values : %s" % len(self.df))
        if(self.count_duplicated_values() > 0):
            self.df.drop_duplicates(inplace=True)
        print("\nRows number after drop duplicated values : %s" % len(self.df))
        return self.df

    def display_missing_values(self): 
        """Display missing values
        """
        for col in self.df.columns.tolist():          
            print("{} column missing values : {}".format(col, self.df[col].isnull().sum()))

    def drop_missing_values(self):
        """Drop missing values
        Returns:
            df (DataFrame): return dataframe without missing values
        """
        self.display_missing_values()

        calc_null = [(col, self.df[col].isna().mean()*100) for col in self.df]
        calc_null = pd.DataFrame(calc_null, columns=["Feature", "Percent NULL"])
        print(calc_null.sort_values("Percent NULL", ascending=False))

        calc_null = calc_null[calc_null["Percent NULL"] > self.percent]
        print(calc_null.sort_values("Percent NULL", ascending=False))
        print(calc_null.count())

        list_calc_null = list(calc_null["Feature"])
        print("\nNumber features before droping when %s percent of values are NULL : %s" % (self.percent, len(self.df.columns)))
        self.df.drop(list_calc_null, axis=1, inplace=True)
        print("\nNumber features after droping when %s percent of values are NULL : %s" % (self.percent, len(self.df.columns)))
        return self.df

    def impute_numeric_features(self):
        """Impute numerical features missing values
        Returns:
            df (DataFrame): return dataframe with numerical features imputed
        """
        print("\nPerforming numeric features imputation ("+self.num_imput+")")

        df_num = self.df.select_dtypes(include=["number"])

        match self.num_imput:
            case 'mean':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
            case 'median':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            case 'mode':
                for col in df_num.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
            case _:
                print("\nWarning : select another method !")

        return self.df
                   
    def impute_categorical_features(self):
        """Impute categorical features missing values
        Returns:
            df (DataFrame): return dataframe with categorical features imputed
        """
        print("\nPerforming categorical features imputation ("+self.cat_imput+")")

        df_cat = self.df.select_dtypes(include=["object"])

        match self.cat_imput:
            case 'mode':
                for col in df_cat.columns.tolist():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
            case _:
                print("\nWarning : select another method !")

        return self.df
             
    def impute_missing_values(self):
        """Impute all features missing values
        Returns:
            df (DataFrame): return dataframe with all features imputed
        """
        self.impute_numeric_features()
        self.impute_categorical_features()
        self.display_missing_values()
        return self.df

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
    
    def get_feature_info(self, feature):
        """Get features information
        Args:
            feature (Dataframe) : feature to extract information
        Returns:
            info (DataFrame): return information
        """
        info = None

        match self.num_imput:
            case 'mean':
                print('mean')
                info = feature.mean()
            case 'median':
                info = feature.median()
            case 'mode':
                info = feature.mode().iloc[0]
            case _:
                print("\nWarning : select another method !")

        print(info)
        return info
        
    def preprocessing(self):
        """Preprocess dataframe
        Returns:
            df (DataFrame): return preprocessed dataframe
        """
        self.drop_duplicated_values()

        self.drop_missing_values()

        self.impute_missing_values()

        self.categorical_features_encoding()
    
        return self.df

