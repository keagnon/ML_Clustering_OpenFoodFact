from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def encode_data(data, column_name):
    data = data.copy()
    le = LabelEncoder()
    data.loc[:, f'{column_name}_encoded'] = le.fit_transform(data[column_name].astype(str))
    return data

def process_nutriscore_grade(data):
    return encode_data(data, 'nutriscore_grade')

def process_main_category_en(data):
    return encode_data(data, 'main_category_en')

def run(data):
    if 'nutriscore_grade' in data.columns:
        data = process_nutriscore_grade(data)
    if 'main_category_en' in data.columns:
        data = process_main_category_en(data)
    return data
