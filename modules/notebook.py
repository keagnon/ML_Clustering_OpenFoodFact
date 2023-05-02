#Data process
import pandas as pd
import itertools
import numpy as np
import data_loader

# Text process
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import UnigramTagger,pos_tag
from nltk import bigrams, ngrams
import re


# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

def list_counter(lst):
    serie = pd.Series(lst)
    return serie.value_counts()

def tokenize(x):
    try :
        return word_tokenize(x)
    except:
        return None
def frequency_list(lst):
    counts = list_counter(lst)
    total = len(lst)
    freq = np.array(counts) / total
    freg = freq * 100
    return pd.Series(freg.round(2), counts.index)

def print_top_10(df, col):
    res = df[col].head(20)
    for el in res:
        print(f'    - {el}')

file_path = "../data/en.openfoodfacts.org.products.csv"
df  = data_loader.get_data(file_path, nrows=1000)

col_names=[]
for x in df.columns:
    col_names.append(x)


index=0
for i in (df.loc[6]):
    print(str(col_names[index])+" : "+str(i))
    index+=1

id_cols = ['code','product_name']

df.head()

for i in range(len(df.columns)) :
    print(df.dtypes.index[i], ' : ', df.dtypes[i])

all_nutrition_cols = [x for x in col_names if '_100g' in x]

print('Pourcentage de valeurs Null par colonnes :\n')

i = 0

nutrition_col_to_keep = []

for col in all_nutrition_cols:
    res = (df[col].isnull().sum() / len(df)) * 100
    res = round(res, 2)
    if res < 70:
        nutrition_col_to_keep.append(col)
        i += 1
        print(f'   - {col} : {res}%')

print("\nNombre colonnes : ", i)

nutrition_col_to_keep

viz1=pd.DataFrame(df.iloc[:,10:].mean().sort_values(ascending=False),columns=['Mean value'])
viz1.style.background_gradient(cmap=sns.light_palette("red", as_cmap=True))

ingredients_col = ['ingredients_text']

print('Pourcentage de valeurs Null par colonnes :\n')

for col in ingredients_col:
    res = (df[col].isnull().sum() / len(df)) * 100
    res = round(res,2)
    print(f'   - {col} : {res}%')

head_ing = df[~df['ingredients_text'].isnull()]['ingredients_text'].head(10)

for el in head_ing:
    print(el)
    print()

[x for x in df.columns if 'catego' in x]

categories_col = ['categories', 'categories_tags', 'categories_en','main_category','main_category_en']


print('Pourcentage de valeurs Null par colonnes :\n')

for col in categories_col:
    res = (df[col].isnull().sum() / len(df)) * 100
    res = round(res,2)
    print(f'   - {col} : {res}%')


main_cate_df=df['main_category_en'].dropna()

main_cate_df

text=main_cate_df.tolist()

text = ' '.join(text).lower()

text=text.replace('fr:','')

lemmatizer = WordNetLemmatizer()

res=word_tokenize(text)

res=[lemmatizer.lemmatize(word) for word in res]

res=' '.join(res).lower()

wordcloud = WordCloud(stopwords = STOPWORDS,
                      collocations=False).generate(res)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilInear')
plt.axis('off')
plt.show()

cols_to_keep = id_cols + categories_col + ingredients_col + nutrition_col_to_keep

print('Pourcentage de valeurs Null par colonnes :\n')
for col in cols_to_keep:
    res = (df[col].isnull().sum() / len(df)) * 100
    res = round(res,2)
    print(f'   - {col} : {res}%')


final_df = df[cols_to_keep]

final_df.columns[8:]

final_df['valeurs_nutritionnelles_list']=pd.Series(dtype='str')

for i in range(len(final_df)):
    val_nutri = ''
    for idx_col in range(8, len(final_df.loc[0])):
        if str(final_df.loc[i][idx_col]) != 'nan':
            val_nutri += (final_df.columns[idx_col] + " " + str(final_df.loc[i][idx_col]) + " ")
    final_df['valeurs_nutritionnelles_list'][i] = val_nutri

print("Product name : " + df['product_name'][849996])
print("Liste nutritionnelle : " + df['valeurs_nutritionnelles_list'][849996])

df_ingrdnnts = final_df[['ingredients_text']]
df_ingrdnnts = df_ingrdnnts[~df_ingrdnnts['ingredients_text'].isnull()]
df_ingrdnnts['ingredients_text'] = df_ingrdnnts['ingredients_text'].astype('str')
spltd_ingrd = df_ingrdnnts['ingredients_text'].apply(lambda x : x.split(','))
spltd_ingrd

cleaned_list_ingr = list(itertools.chain.from_iterable(spltd_ingrd))
cleaned_list_ingr = pd.read_csv('comma_split_ingrd.csv')
first_count = list_counter(cleaned_list_ingr)

tokenised_ingr = list(df_ingrdnnts['ingredients_text'].apply(tokenize))
tokenised_ingr = [str(x).lower() for x in tokenised_ingr]
tokenised_without_specials = [x for x in tokenised_ingr if re.match('[a-zA-Z]+',str(x))]
all_stopwords = stopwords.words()
tokenised_without_stopwords = [x for x in tokenised_without_specials if x not in all_stopwords]
lemmetized_tokens = [lemmatizer.lemmatize(str(word)) for word in tokenised_without_stopwords]
print(f"""
Tous les tokes : {len(tokenised_ingr)} élements

Tokens sans charctères spéciaux : {len(tokenised_without_specials)} élements

Tokens sans Stop words  {len(tokenised_without_stopwords)} élements""")

counts_tokenised_1gram = list_counter(lemmetized_tokens)
freq_tokenised_1gram = frequency_list(lemmetized_tokens)
for i in range(10000, 250000, 20000):
    res = sum(counts_tokenised_1gram > i)
    print(f"Nombre d'éléments cités plus de {i} fois : {res}\n")

for i in np.arange(0.2, 3, 0.2):
    i = round(i,1)
    res = sum(freq_tokenised_1gram > i)
    print(f"Nombre d'éléments ayant une fréquence supérieur à {i}% : {res}\n")

top100_ingr = list(counts_tokenised_1gram.index[:100])

print('Les 100 éléments les plus cités :\n\n')

i = 0
res = ''
for el in top100_ingr :
    i+=1
    res = res + el + '   '
    if i%10 == 0:
        print(res, '\n')
        res =''

res=' '.join(top100_ingr).lower()

wordcloud = WordCloud(stopwords = STOPWORDS,
                      collocations=False).generate(res)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilInear')
plt.axis('off')
plt.show()