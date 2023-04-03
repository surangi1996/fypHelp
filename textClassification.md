# Multinomial Naїve Bayes’ For Documents Classification and Natural Language Processing (NLP) in details


Multinomial Naïve Bayes is a probabilistic machine learning algorithm that is commonly used for document classification and natural language processing (NLP). 
The algorithm is based on Bayes' theorem and assumes that the features are conditionally independent given the class label. 
This makes it particularly suitable for text classification tasks where the feature space is high-dimensional and sparse.

In NLP, the goal of document classification is to assign one or more labels to a document based on its content. 
For example, given a set of product reviews, we may want to classify each review as positive or negative. 
To do this, we need to preprocess the text data by tokenizing the text into words, removing stop words, and converting the text into numerical features that can be used as input to a machine learning algorithm.

The first step in implementing a Multinomial Naïve Bayes model is to convert the text data into numerical features. 
This is typically done using the bag-of-words model, which represents each document as a vector of word counts. 
The bag-of-words model ignores the order of the words and treats each word as a separate feature. 
To do this, we use a CountVectorizer object from scikit-learn, which transforms the text data into a matrix of token counts.

Once the text data has been converted into numerical features, we can train a Multinomial Naïve Bayes classifier on the training data. 
The Multinomial Naïve Bayes classifier is a probabilistic classifier that uses Bayes' theorem to calculate the probability of a document belonging to a particular class label given its features. 
The classifier assumes that the features are conditionally independent given the class label, which means that it calculates the probability of a document belonging to a particular class label by multiplying the probabilities of each feature given the class label.

To make a prediction for a new document, we simply use the trained Multinomial Naïve Bayes classifier to calculate the probability of the document belonging to each class label, and then choose the class label with the highest probability. 
This process is repeated for each document in the test data, and the predicted class labels are compared to the true class labels to evaluate the performance of the classifier.

Overall, Multinomial Naïve Bayes is a simple and effective algorithm for document classification and natural language processing tasks. 
Its simplicity makes it particularly suitable for high-dimensional and sparse feature spaces, such as those commonly encountered in NLP. 
However, it may not always be the best choice for more complex classification tasks, and more advanced techniques may be required.

here's an example implementation of Multinomial Naïve Bayes for document classification and natural language processing (NLP) using Python, NumPy, and scikit-learn:

# example 01

```
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data
train_data = [
    ('this is a positive example', 'positive'),
    ('this is a negative example', 'negative'),
    ('I love this product', 'positive'),
    ('I hate this product', 'negative'),
    ('I recommend this product', 'positive'),
    ('I do not recommend this product', 'negative')
]

test_data = [
    'this is a test example',
    'I am not sure about this product'
]

# Convert the data into numerical features using the CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([text for text, label in train_data])
y_train = np.array([label for text, label in train_data])
X_test = vectorizer.transform(test_data)

# Train the Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the predictions
for i, text in enumerate(test_data):
    print(f'Text: {text}  Prediction: {y_pred[i]}')
```
# example 02
https://towardsdatascience.com/multinomial-na%C3%AFve-bayes-for-documents-classification-and-natural-language-processing-nlp-e08cc848ce6

```
import os
import re
import csv
import math
import nltk
import numpy as np

def log_p(n):
    # Get the probability's n natural algorithm
    return abs(math.log(abs(n))) \
        if n != 0 and n != 1 else 1

def get_fea_class(D,k):
    # Get all documents in D belonging to the k-th class in C
    return np.array([ d for d in D if k == int(d[0]) ])

def get_count_class(D,k):
    # Get the count p(Ck) of the class Ck in documents D
    return len(get_fea_class(D, k))

def get_counts_term(D,w):
    # Get the count of the term w occurrences in each document from D
    count_wt = np.array([ len([ term \
        for term in d[1] if w == term ]) for d in D ])
    # Get the total count of documents from D, containing the term w
    return len(np.array([ f_wt \
        for f_wt in count_wt if f_wt > 0 ]))

def get_prob_class(D,k):
    # Get the probability p(Ck) of the k-th class Ck
    return get_count_class(D,k) / len(D)

def get_probs_term(D,w):
    # Get the probability of the term w occurrence 
    # in each document from the class Ck
    return get_counts_term(D,w) / len(D)

def parse(S):
    W = S.lower().split()

    # Parse the string S, performing 
    # the normalization and word-stamming using NLTK library
    W = np.array([ re.sub(r"""[,.;@#?!&$\']+\ *""", '', w) for w in W])
    W = np.array([ tag[0] for tag in nltk.pos_tag(W) \
        if re.match('NN', tag[1]) != None or re.match('JJ', tag[1]) != None ])

    return np.array([ w for w in W if len(w) > 2 ])
    
def build_model(D):
    # Build the class prediction model, 
    # based on the corpus of documents in D
    D = np.array([ np.array([ d[0], parse(d[1]) ], \
        dtype=object) for d in D ], dtype=object)
    return np.array([ d for d in D if len(d[1]) > 0 ])

def compute(D,C,S):
    W = parse(S);                 # A set of terms W in the sample S
    Pr = np.empty(0);             # A set of posteriors Pr(Ck | W)

    n = len(W); m = len(C)        # n - # of terms W in S
                                  # m - # of classes in C

    # For each k-th class Ck, compute the posterior Pr(Ck | W)
    for k in range(m):
        pr_ck_w = 0                  # pr_ck_w - the likelihood P(Ck | wi) 
                                     # of Ck is the class of the term wi

        d_ck = get_fea_class(D,k)    # d_ck - A set of documents from the class Ck
        p_ck = get_prob_class(D,k)   # p_ck - Probability of the k-th class Ck in documents D

        # For each term W[i], compute the likelihood P(Ck | wi)
        for i in range(n):
            # Obtain the count and probability of the 
            # term W[i] in the documents from class Ck
            prob_wd_n = get_probs_term(d_ck, W[i])
            count_wt_n = get_counts_term(d_ck, W[i])
            
            pr_ck_w += count_wt_n * \
                log_p(prob_wd_n) if count_wt_n > 0 else 0

        pr_ck_w += p_ck

        # Append the posterior Pr(Ck | W) of the class Ck to the array Pr
        Pr = np.append(Pr, pr_ck_w)

    # Obtain an index of the class Cs as the class in C, 
    # having the maximum posterior Pr(Ck | W)
    Cs = np.where(Pr == np.max(Pr))[0][0]
   
    return Pr,Cs   # Return the array of posteriors Pr
                   # and the index of sample S class Cs

def evaluate(T,D,C):
    print('Classification:')
    print('===============\n')

    # For each sample S in the set T, compute the class of S
    # Estimate the real classification's multinomial entropy and its expectation
    for s in T[:,1]:
        pr_s = '\0'; \
            Pr,Cs = compute(D,C,s)
        for ci,p in zip(range(len(C)),Pr):
            pr_s += prob_stats % (C[ci][1],p)

        print(sampl_stats % (s, C[Cs][1] \
            if np.sum(Pr) > 0 else 'None', pr_s))
```
