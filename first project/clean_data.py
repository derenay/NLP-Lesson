import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec




lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

df = pd.read_csv(r'first project/data/spam_dataset.csv')

df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

token_words = [word_tokenize(word) for word in df['text']]


upper_list=[]

for words in token_words:
    new_list = []
    for word in words:
        if word not in stop_words:
            new_list.append(lemmatizer.lemmatize(word))
    upper_list.append(new_list)


df['cleaned_words'] = upper_list

word2vec_model = Word2Vec(sentences=upper_list, vector_size=50, window=5, min_count=1, sg=1, epochs=100)
word_vectors = word2vec_model.wv

# Function to convert sentences to vectors (by averaging word embeddings)
def get_sentence_vector(words):
    valid_words = [word for word in words if word in word_vectors]
    if len(valid_words) == 0:
        return np.zeros(50)  # Return a zero vector if no valid words exist
    return np.mean([word_vectors[word] for word in valid_words], axis=0)


df['sentence_embedding'] = df['cleaned_words'].apply(get_sentence_vector)

df['label'] = df['label']  # No change here

#df.to_csv('cleaned_and_featured_data.csv', index=False)

print(df[['text', 'cleaned_words', 'label', 'sentence_embedding']])


