# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:38:27 2023

@author: fredh
"""





"""
Data Processing 
"""

    # Import the data 

import os 
import pickle

PATH = os.getcwd()

path1 = os.path.join(PATH, 'translated_1700_fa_citation.pkl')
path2 = os.path.join(PATH, 'translated_1700_fa_no_citation.pkl')

path3 = os.path.join(PATH, 'translated_1700_cn_citation.pkl')
path4 = os.path.join(PATH, 'translated_630_cn_no_citation.pkl')
path5 = os.path.join(PATH, 'translated_1070_cn_no_citation.pkl')

path6 = os.path.join(PATH, 'translated_1700_rdm_citation.pkl')
path7 = os.path.join(PATH, 'translated_1700_rdm_no_citation.pkl')


with open(path1, 'rb') as f:
    fa_citation = pickle.load(f)

with open(path2, 'rb') as f:
    fa_no_citation = pickle.load(f)
fa_no_citation = fa_no_citation[1:]             # on enlève les titres


with open(path3, 'rb') as f:
    cn_citation = pickle.load(f)

with open(path4, 'rb') as f:
    cn_no_citation_1 = pickle.load(f)
    
with open(path5, 'rb') as f:
    cn_no_citation_2 = pickle.load(f)



with open(path6, 'rb') as f:
    rdm_citation = pickle.load(f)

with open(path7, 'rb') as f:
    rdm_no_citation = pickle.load(f)


cn_no_citation = cn_no_citation_1 + cn_no_citation_2


# print(len(fa_citation))
# print(len(fa_no_citation))
# print(len(cn_citation))
# print(len(cn_no_citation))
# print(len(rdm_citation))
# print(len(rdm_no_citation))


    # Extract and join data
citation_all = fa_citation + cn_citation + rdm_citation
no_citation_all = fa_no_citation + cn_no_citation + rdm_no_citation

citations = [citation_all[k][8] for k in range(len(citation_all))]
no_citations = [no_citation_all[k][8] for k in range(len(no_citation_all))]


# OK bizarre les citations et tout ... (bon j'ai vérifié rapidement à la main et on va dire que c'est bon)







"""
TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import nltk

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Download French stopwords
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

# Load data
X = citations + no_citations 
y = [1]*len(citations) + [0]*len(no_citations)          # label 1 si citation, 0 sinon

# Tokenize text using TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
X = [' '.join(tokenizer.tokenize(text)) for text in X]

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words=french_stopwords)
X_tfidf = tfidf.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

# Evaluate classifier on testing set
accuracy = accuracy_score(y_test, y_pred_clf)
print(f"Accuracy: {accuracy:.2f}")
# >>> Accuracy: 0.63

print(confusion_matrix(y_test, y_pred_clf))
# >>> [[800 203]
#      [543 494]]




# Avec une régression logistique : 
from sklearn.linear_model import LogisticRegression


# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# >> Accuracy : 0.66

print(confusion_matrix(y_test, y_pred))
# >> [[644 359]
#     [341 696]]







# See the weigths of the models (to see the correlation between certain words and the classification)
# Logistic regression


# Get feature names from the TF-IDF vectorizer
feature_names = tfidf.get_feature_names()

# Get the weight vector from the logistic regression model
weights = lr.coef_[0]

# Create a list of (feature, weight) pairs
feat_weights = list(zip(feature_names, weights))

# Sort the list by weight (ascending order)
feat_weights.sort(key=lambda x: x[1])

# Print the top n features with the highest and lowest weights
n = 10
print(f"Top {n} features with highest weights:")
for feat, weight in feat_weights[-n:]:
    print(f"{feat}: {weight:.2f}")
print(f"\nTop {n} features with lowest weights:")
for feat, weight in feat_weights[:n]:
    print(f"{feat}: {weight:.2f}")
    
    
# shit j'ai eu des problèmes de traduction (cf. les 10 mots avec les poids les plus élevés)

"""
Top 10 features with highest weights:
2014: 1.21
2015: 1.22
the: 1.23
annoncé: 1.25
to: 1.26
faisant: 1.36
and: 1.40
tout: 1.50
selon: 2.08
déclaré: 2.83

Top 10 features with lowest weights:
né: -3.18
peuvent: -2.17
mort: -2.14
généralement: -1.93
également: -1.90
depuis: -1.85
ville: -1.78
remporté: -1.68
ensuite: -1.67
joué: -1.66
"""