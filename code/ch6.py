"""Train word embeddings on `corpus` using the word2vec (binary logistic
regression) skipgram approach. You may use numpy or sklearn. If you get stuck,
just use gensim.models.Word2Vec
(https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)
(pip install gensim) to train embeddings to complete the remaining steps of the
assignment.

    Vector length: 10
    Iterations: 20
    Negative sample size: 1
    Window size: 5 (target +- 2)

Write a function to compute the cosine similarity between two vectors.

Then compute a cosine similarity matrix (vocabulary used as both columns and
rows). Which 10 words are closest to "dogs"? Determine why "mammals" and
"humans" are close to "dogs", even though they do no co-occur in the window
(You can change words in the corpus and re-compute to test your theories).
"""

corpus = [['cats', 'and', 'dogs', 'are', 'pets'],
          ['humans', 'have', 'had', 'pets', 'for', 'a', 'long', 'time',
           'right', 'from', 'their', 'nomadic', 'days', 'of', 'existence'],
          ['dogs', 'have', 'their', 'genetic', 'roots', 'in', 'wolves'],
          ['the', 'large', 'variety', 'of', 'dogs', 'are', 'largely', 'the',
           'results', 'of', 'humans', 'performing', 'selective', 'breeding'],
          ['cats', 'are', 'felines', 'like', 'tigers'],
          ['cats', 'are', 'mammals']]
