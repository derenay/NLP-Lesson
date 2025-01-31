from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize



nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

text = "Bu bir test cümlesidir ve gereksiz kelimeleri çıkarmalıyız."
tokens = word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)  # Stopwords temizlenmiş kelimeler
