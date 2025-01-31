import pandas as pd
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re


nltk.download("punkt")  # Ensure tokenization works

# Manually labeled dataset (training data)
labeled_data = pd.DataFrame({
    "text": [
        "I love this product, it's amazing!",
        "Absolutely terrible, I hate it.",
        "The food was okay, but nothing special.",
        "This is the worst movie I have ever seen!",
        "Great customer service, very friendly staff."
    ],
    "label": ["Positive", "Negative", "Neutral", "Negative", "Positive"]
})



tokenized_sentences  = [word_tokenize(re.sub(r'[^\w\s]', '', text.lower())) for text in labeled_data['text']]



ord2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)









