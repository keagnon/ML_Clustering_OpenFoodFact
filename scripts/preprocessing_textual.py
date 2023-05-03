from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_textual_data(data, column):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data[column].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names())

    # Combine the original dataframe with the new TF-IDF dataframe
    combined_data = pd.concat([data, tfidf_df], axis=1)

    return combined_data

def run(data):
    # Your existing preprocessing steps

    # Additional processing for textual data
    textual_column = 'ingredients_text'
    data = preprocess_textual_data(data, textual_column)

    return data