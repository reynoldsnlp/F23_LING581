"""Exercises from ch 4 of Jurafsky and Martin Speech and Language Processing v3"""

from collections import Counter
from collections import defaultdict
from math import prod
from pprint import pprint

from numpy import argmax

"""
4.1
Assume the following likelihoods for each word being part of a positive or
negative movie review, and equal prior probabilities for each class.
  pos neg
I 0.09 0.16
always 0.07 0.06
like 0.29 0.06
foreign 0.04 0.15
films 0.08 0.11
What class will Naive bayes assign to the sentence “I always like foreign
films.”?
"""
print('4.1')
labels = ('pos', 'neg')
likelihoods = {'I': {'pos': 0.09, 'neg': 0.16},
               'always': {'pos': 0.07, 'neg': 0.06},
               'like': {'pos': 0.29, 'neg': 0.06},
               'foreign': {'pos': 0.04, 'neg': 0.15},
               'films': {'pos': 0.08, 'neg': 0.11}}
text = ['I', 'always', 'like', 'foreign', 'films']  # No likelihood given for '.'
label_probs = []
for label in labels:
    tok_likelihoods = [likelihoods[token][label] for token in text]
    print(f"Token likelihoods for {label}:", tok_likelihoods)
    label_prob = prod(tok_likelihoods)
    label_probs.append(label_prob)
print("Label probabilities:", dict(zip(labels, label_probs)))
print("Most likely label:", labels[argmax(label_probs)])
print("\n\n\n")

"""
4.2
Given the following short movie reviews, each labeled with a genre, either
comedy or action:
1. fun, couple, love, love comedy
2. fast, furious, shoot action
3. couple, fly, fast, fun, fun comedy
4. furious, shoot, shoot, fun action
5. fly, fast, shoot, love action
and a new document D:
fast, couple, shoot, fly
compute the most likely class for D. Assume a naive Bayes classifier and use
add-1 smoothing for the likelihoods.
"""
print('4.2')
corpus = [(['fun', 'couple', 'love', 'love'], 'comedy'),
          (['fast', 'furious', 'shoot'], 'action'),
          (['couple', 'fly', 'fast', 'fun', 'fun'], 'comedy'),
          (['furious', 'shoot', 'shoot', 'fun'], 'action'),
          (['fly', 'fast', 'shoot', 'love'], 'action')]
num_docs = len(corpus)
vocab = set()
for tokens, label in corpus:
    for token in tokens:
        vocab.add(token)
vocab_size = len(vocab)
label_freqs = Counter(label for tokens, label in corpus)
priors = {label: label_count / num_docs
          for label, label_count in label_freqs.items()}
print("priors:", priors)

freqs_by_label = defaultdict(Counter)
for tokens, label in corpus:
    freqs_by_label[label].update(tokens)


likelihoods = defaultdict(dict)
for label, freqs in freqs_by_label.items():
    N = sum(freqs.values())
    for token in vocab:
        freq = freqs[token]
        likelihoods[token][label] = (freq + 1) / (N + vocab_size)
print('Likelihoods:')
pprint(likelihoods)

new_doc = ['fast', 'couple', 'shoot', 'fly']
labels = list(label_freqs)
label_probs = []
for label in labels:
    new_doc_likelihoods = [likelihoods[token][label] for token in new_doc]
    print(f"Token likelihoods for '{label}':", new_doc_likelihoods)
    label_prob = prod(new_doc_likelihoods) * priors[label]
    label_probs.append(label_prob)
print("Document:", new_doc)
print("Label probabilities:", dict(zip(labels, label_probs)))
print("Most likely label:", labels[argmax(label_probs)])
print("\n\n\n")


"""
4.3
Train two models, multinomial naive Bayes and binarized naive Bayes, both
with add-1 smoothing, on the following document counts for key sentiment
words, with positive or negative class assigned as noted.
doc “good” “poor” “great” (class)
d1. 3 0 3 pos
d2. 0 1 2 pos
d3. 1 3 0 neg
d4. 1 5 2 neg
d5. 0 2 0 neg
Use both naive Bayes models to assign a class (pos or neg) to this sentence:
A good, good plot and great characters, but poor acting.
Recall from page 6 that with naive Bayes text classification, we simply ignore
(throw out) any word that never occurred in the training document. (We don’t
throw out words that appear in some classes but not others; that’s what addone smoothing is for.) Do the two models agree or disagree?
"""
print('4.3')

