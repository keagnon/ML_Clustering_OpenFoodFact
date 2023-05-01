import matplotlib.pyplot as plt
import multidict as multidict#a installer pour env
from wordcloud import WordCloud
import re
from stop_words import get_stop_words #a installer pour env
from unidecode import unidecode#a installer pour env

def explode_col_by_delimiter(df, 
                             column, 
                             delimiter=',', 
                             new_col_name="texts", 
                             id_col="code", 
                             drop_col=True, 
                             lower=True,
                             nan_to_string=True,
                             clean_stop_words=True
                             ):
    """Explode a df column character separated strings 
    Args:
        df (DataFrame): dataframe to work on
        column (string): column to work on
        delimiter (string, optional): delimiter used to explode a row string into multiple rows
            By default ,
        new_col_name (string, optional): result column name
            By default texts
        id_col (string, optional): column to use to keep an id on rows
            By default code
        drop_col (boolean, optional): should the column which we are working on be dropped ?
            By default True
        lower (boolean, optional): text automatically lowered
            By default True
        nan_to_string (boolean, optional): NaN automatically casted to string
            By default True
        clean_stop_words (boolean, optional): Stop words are automatically removed
            By default True
    Returns:
        dataframe: output dataframe
    """
    columns = [id_col]
    columns.append(column)
    new_df = df.loc[:, columns]
    new_df[new_col_name] = df[column].str.split(delimiter).explode().reset_index(drop=True)
    if drop_col:
        new_df.drop(column, axis=1, inplace=True)
    if lower:
        new_df[new_col_name] = new_df[new_col_name].str.lower()
    if nan_to_string:
        new_df[new_col_name].fillna("NaN", inplace=True)
    if clean_stop_words:
        new_df = rid_of_stop_words(new_df, col_to_test=new_col_name)

    new_df[new_col_name] = new_df[new_col_name].str.strip()
    return new_df

def exclude_words(df, words, col_to_test="texts"):
    """Exclude dataframe rows when words provided are found
    Args:
        df (DataFrame): dataframe to work on
        words (array string): words to exclude from dataframe entries
        col_to_test (string, optional): column to work on
            By default texts
    Returns:
        dataframe: output dataframe
    """
    return df[~df[col_to_test].isin(words)]

def display_word_cloud(text, column, max_words=200, min_font_size=6,
                       width=2000, height=1000, collocations=False, mask=None):
    """Display basic word cloud from string
    Args:
        text (string): string to use for the word cloud
        column (string): column name to viz
        args (WordCloud args): args provided to configurate the WordCloud: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
    Returns:
    """
    cloud = WordCloud(background_color="white", max_words=max_words, mask=mask, 
                      min_font_size=min_font_size, width=width, height=height, 
                      collocations=collocations)
    cloud.generate(text)
    plt.axis("off")
    plt.imshow(cloud, interpolation="bilinear")
    plt.title("Word cloud of " + column + " feature")
    plt.show()
    
def display_word_cloud_frequencies(dicti, column, max_words=200, min_font_size=6, 
                                   width=2000, height=1000,
                                   collocations=False, mask=None):
    """Display basic word cloud from multidict
    Args:
        dicti (MultiDict): dictionnary with frequencies of words
        column (string): column name to viz
        args (WordCloud args): args provided to configurate the WordCloud: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
    Returns:
    """
    cloud = WordCloud(background_color="white", max_words=max_words, 
                      mask=mask, min_font_size=min_font_size, 
                      width=width, height=height, collocations=collocations)
    cloud.generate_from_frequencies(dicti)
    plt.axis("off")
    plt.imshow(cloud, interpolation="bilinear")
    plt.title("Word cloud of " + column + " feature")
    plt.show()
    
def get_frequency_dict_from_text(sentence):
    """Get words frequency from string
    Args:
        sentence (string): string to use
    Returns:
        MultiDict: dictionnary of word frequencies
    """
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict

def rid_of_stop_words(df, language='french', col_to_test="texts"):
    """Remove stop words from dataframe specific column entries
    Args:
        df (DataFrame): dataframe to use
        language (string): stop words language 
            By default french
        col_to_test (string): column to use
            By default texts
    Returns:
        DataFrame: dataframe without stopwords
    """
    stop_words = get_stop_words(language=language)
    return df[~df[col_to_test].isin(stop_words)]

def normalize_text(text):
    """Normalize string
    Args:
        text (string): string to use
    Returns:
        String: text normalized
    """
    return unidecode(text)

