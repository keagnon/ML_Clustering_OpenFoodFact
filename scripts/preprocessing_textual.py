import pandas as pd
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


def process_additives_tags(data):
    data = data.copy()

    # Remove the part before the colon in the tags
    data['additives_tags'] = data['additives_tags'].apply(
        lambda x: ','.join(tag.split(':')[-1] for tag in str(x).split(',')))

    vectorizer = TfidfVectorizer(token_pattern=r'[a-z0-9]+')
    additives_tfidf = vectorizer.fit_transform(data['additives_tags'].fillna(''))

    # Create a DataFrame with the TfidfVectorizer results
    additives_df = pd.DataFrame(additives_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the original data with the additives_tags features
    data = pd.concat([data, additives_df], axis=1)
    return data


def run(data):
    if 'nutriscore_grade' in data.columns:
        data = process_nutriscore_grade(data)
    if 'main_category_en' in data.columns:
        data = process_main_category_en(data)
    #if 'additives_tags' in data.columns:
        #data = process_additives_tags(data)
    return data
