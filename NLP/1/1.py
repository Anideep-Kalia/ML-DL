pip install nltk
paragraph ="""+++"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline



# Tokenization
nltk.download('punkt')
sentences=nltk.sent_tokenize(paragraph)

stemmer=PorterStemmer()
stemmer.stem('going')

lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize('history')


corpus=[]
# removing all the numbers and special characters i.e. except alphabets everything is removed and converted to small case
for i in range(len(sentences)):
    review= re.sub('[^a-zA-Z]',' ',sentences[i])
    review =review.lower()
    review.split()
    corpus.append(review)

for i in corpus:
    print(i)

# stemmming, tokenizing & removing stopwords
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))
            
# lemmatization
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))

cv=CountVectorizer()

X=cv.fit_transform(corpus)
cv.vocabulary_
corpus[0]
X[0].toarray()


# TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
tfidf.get_feature_names_out()
X[0]

# N-gram Demo
cv = CountVectorizer(ngram_range=(2,3))
X = cv.fit_transform(corpus)
cv.get_feature_names_out()

# max_features
cv = CountVectorizer(max_features=10)
X = cv.fit_transform(corpus).toarray()

# tfidf and LogisticRegression
# TfidfVectorizer(sublinear_tf=True) -> sublinear_tf
pipe = Pipeline([
  ('tfidf', TfidfVectorizer()),
  ('clf', LogisticRegression(max_iter=1000))
])
param_grid = {
  'tfidf__min_df': [1,2,5],
  'tfidf__max_df': [0.9, 0.95, 0.99],
  'tfidf__max_features': [5000, 10000, None],
  'clf__C': [0.1, 1, 10]
}
gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
gs.fit(X_text, y)










