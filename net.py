import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataCleaner:
    def __init__(self, df, num_imput='mean', cat_encoding='label'):
        self.df = df
        self.num_imput = num_imput
        self.cat_encoding = cat_encoding
        
    def count_duplicated_values(self):
        """Compte les valeurs dupliquées pour chaque caractéristique
        Retourne :
            df (DataFrame) : retourne la somme des valeurs dupliquées """
        return self.df.duplicated().sum()
    
    def drop_duplicated_values(self):
        """Supprime les valeurs dupliquées
        Retourne :
            df (DataFrame) : retourne le dataframe sans les valeurs dupliquées """
        print("\nNombre de lignes avant la suppression des valeurs dupliquées : %s" % len(self.df))
        if(self.count_duplicated_values() > 0):
            self.df.drop_duplicates(inplace=True)
        print("\nNombre de lignes après la suppression des valeurs dupliquées : %s" % len(self.df))
        return self.df
    
    def impute_missing_values(self):
        """Remplit les valeurs manquantes pour les variables numériques
        Retourne :
            df (DataFrame) : retourne le dataframe avec les valeurs manquantes remplies """

        if self.num_imput == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif self.num_imput == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif self.num_imput == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)

        print("\nNombre de valeurs manquantes après l'imputation : %s" % self.df.isnull().sum().sum())
        return self.df

  

    def cleaning(self):
        self.drop_duplicated_values()

        self.impute_missing_values()



