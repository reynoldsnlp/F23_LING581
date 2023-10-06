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

from collections import Counter
from collections import defaultdict
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=DeprecationWarning)  # TODO remove

corpus = [['cats', 'and', 'dogs', 'are', 'pets'],
          ['humans', 'have', 'had', 'pets', 'for', 'a', 'long', 'time',
           'right', 'from', 'their', 'nomadic', 'days', 'of', 'existence'],
          ['dogs', 'have', 'their', 'genetic', 'roots', 'in', 'wolves'],
          ['the', 'large', 'variety', 'of', 'dogs', 'are', 'largely', 'the',
           'results', 'of', 'humans', 'performing', 'selective', 'breeding'],
          ['cats', 'are', 'felines', 'like', 'tigers'],
          ['cats', 'are', 'mammals']]


def init_vocab(flat_corpus):
    """Create a vocabulary mapping tokens to integers.

    Parameters
    ----------
    flat_corpus : List[str]
        Corpus tokenized without sentences
    """
    word2index = {}
    index2word = []
    i = 0
    for tok in flat_corpus:
        if tok not in word2index:
            word2index[tok] = i
            index2word.append(tok)
            i += 1
    return word2index, index2word


def get_windows(corpus, window):
    """Yield windows and the index of the target word in that window.

    Parameters
    ----------
    corpus : List[int]
        Flattened list of word indices
    window : int
        Maximum distance from target word to include in the window
    """
    max_i = len(corpus) - 1
    for trgt_i, tok in enumerate(corpus):
        trgt_i_win = window
        win_start = trgt_i - window
        win_end = trgt_i + window + 1
        if win_start < 0:
            trgt_i_win = window + win_start
            win_start = None
        elif win_end > max_i:
            win_end = None
        yield corpus[win_start:win_end], trgt_i_win


def get_window_dict(corpus, window=5):
    """Return a dictionary of sets of context words that occur in each
    target word's window.

    Parameters
    ----------
    corpus : List[int]
        Flattened list of word indices
    window : int
        Maximum distance from target word to include in the window
    """
    window_dict = defaultdict(set)
    for window, trgt_i in get_windows(corpus, window=window):
        trgt = window.pop(trgt_i)
        for context_word in window:
            window_dict[trgt].add(context_word)
    return window_dict


def get_batch(corpus, window=5, negative=0, freqs=None, window_dict=None):
    """Batch of training data.

    Parameters
    ----------
    corpus : List[str]
        Word-tokenized corpus
    window : int
        Maximum distance between the current and predicted word within a
        sentence.
    negative : int
        If > 0, negative sampling will be used, the int for negative specifies
        how many “noise words” should be drawn (usually between 5-20). If set
        to 0, no negative sampling is used.
    freqs : Dict[int, int]
        Weighted MLE frequency distribution. Required if `negative` is not 0.
    window_dict : Dict[int, Set[int]]
        Dictionary of which words occur in the context of each target word.
    """
    freq_words, freq_weights = zip(*freqs.items())

    training_data = []
    for window, trgt_i in get_windows(corpus, window=window):
        target_word = window.pop(trgt_i)
        context_words = window
        for c_word in context_words:
            minibatch = []
            minibatch.append(target_word)
            minibatch.append(c_word)
            if negative > 0:
                neg_context_words = set()
                while len(neg_context_words) < negative:
                    choice = random.choices(freq_words, freq_weights)[0]
                    if choice not in window_dict[target_word]: # necessary?
                        neg_context_words.add(choice)
                for c_word in neg_context_words:
                    minibatch.append(c_word)
            training_data.append(minibatch)
    return training_data


def loss(w, c_pos, c_negs):
    """Compute loss. Based on formula (6.34) in SLP textbook."""
    return -(np.log(sigmoid(c_pos.dot(w)))
             + np.sum(np.log(sigmoid(-c_neg.dot(w))) for c_neg in c_negs))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cos_similarity(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def train_sgns(sentences, vector_size=100, window=4, negative=None,
               ns_exponent=0.75, epochs=5, alpha=0.025):
    """Train weights for a SkipGram Negative Sampling model.

    Parameters
    ----------
    sentences : List[List[str]]
        Sentence- and word-tokenized corpus
    vector_size : int
        Length of output word vectors
    window : int
        Maximum distance from target word to include in the window
    negative : int
        If > 0, negative sampling will be used, the int for negative specifies
        how many “noise words” should be drawn (usually between 5-20). If set
        to 0, no negative sampling is used.
    ns_exponent : float
        The exponent used to shape the negative sampling distribution. A value
        of 1.0 samples exactly in proportion to the frequencies, 0.0 samples
        all words equally, while a negative value samples low-frequency words
        more than high-frequency words. The popular default value of 0.75 was
        chosen by the original Word2Vec paper.
    epochs : int
        The number of iterations to train a batch.
    alpha : float
        Learning rate
    """
    flat_corpus = [word for sent in corpus for word in sent]
    print(f"Corpus: {flat_corpus}")
    word2index, index2word = init_vocab(flat_corpus)
    # max_word_len = max(len(w) for w in index2word)  # for formatting output
    # print(f"{word2index=}")
    # print(f"{list(enumerate(index2word))=}")
    # print(f"Vocab (size {len(index2word)}): ", index2word)
    int_corpus = [word2index[word] for sent in corpus for word in sent]

    freqs = Counter(int_corpus)
    N = len(int_corpus)
    freqs = {word: freq / N for word, freq in freqs.items()}
    ns_exponent_denom = sum(p ** ns_exponent
                                  for p in freqs.values())
    freqs = {word: p ** ns_exponent / ns_exponent_denom
             for word, p in freqs.items()}

    window_dict = get_window_dict(int_corpus, window=window)

    w_vectors = np.random.rand(len(index2word), vector_size)
    c_vectors = np.random.rand(len(index2word), vector_size)

    losses = []
    for _ in range(epochs):
        training_data = get_batch(int_corpus, window=window, negative=negative,
                                  freqs=freqs, window_dict=window_dict)
        for minibatch in training_data:
            w_int, c_pos_int, *c_negs_int = minibatch
            w = w_vectors[w_int]
            c_pos = c_vectors[c_pos_int]
            c_negs = [c_vectors[c_neg_int] for c_neg_int in c_negs_int]
            losses.append(loss(w, c_pos, c_negs))
            # print('Target word:', w_int,
            #       f"{index2word[w_int]:<{max_word_len}}",
            #       'Pos context word:', c_pos_int,
            #       f"{index2word[c_pos_int]:<{max_word_len}}",
            #       'Neg context word(s):', c_negs_int,
            #       " ".join(index2word[c] for c in c_negs_int))

            c_updates = {}
            # c_pos update taken from formula (6.38)
            c_updates[c_pos_int] = c_pos - alpha * (sigmoid(c_pos.dot(w)) - 1) * w
            for c_neg_int, c_neg in zip(c_negs_int, c_negs):
                # c_neg_update taken from formula (6.39)
                c_neg_update = c_neg - alpha * sigmoid(c_neg.dot(w)) * w
                c_updates[c_neg_int] = c_neg_update
            # w_update taken from formula (6.40)
            w_update = w - alpha * ((sigmoid(c_pos.dot(w)) - 1) * c_pos
                                    + np.sum((sigmoid(c_neg.dot(w)) * c_neg)
                                             for c_neg in c_negs))

            for c_int, c_update in c_updates.items():
                c_vectors[c_int] = c_update
            w_vectors[w_int] = w_update
    plt.plot(losses)
    plt.title("Loss")
    plt.show()
    return w_vectors, c_vectors, index2word, word2index


if __name__ == "__main__":
    SEED = 314159 
    np.random.seed(SEED)
    random.seed(SEED)
    w_vectors, c_vectors, index2word, word2index = train_sgns(corpus,
                                                              vector_size=10,
                                                              window=2,
                                                              negative=1,
                                                              epochs=20)
    similarity_df = pd.DataFrame([[cos_similarity(w_vectors[i], w_vectors[j])
                                   for j in range(len(w_vectors))]
                                  for i in range(len(w_vectors))],
                                 columns=index2word,
                                 index=index2word)
    most_like_dogs = similarity_df["dogs"].nlargest(10)
    print('"mammals" and "humans" (never co-occur with "dogs") are higher '
          'than "pets" (which does co-occur with "dogs").')
    print(most_like_dogs)

    print('\n\nChange the first word from "cats" to "rocks", and "mammals" is '
          'no longer similar to "dogs".')
    np.random.seed(SEED)
    random.seed(SEED)
    rock_corpus = corpus.copy()
    rock_corpus[0][0] = "rocks"
    w_vectors, c_vectors, index2word, word2index = train_sgns(rock_corpus,
                                                              vector_size=10,
                                                              window=2,
                                                              negative=1,
                                                              epochs=20)
    similarity_df = pd.DataFrame([[cos_similarity(w_vectors[i], w_vectors[j])
                                   for j in range(len(w_vectors))]
                                  for i in range(len(w_vectors))],
                                 columns=index2word,
                                 index=index2word)
    most_like_dogs = similarity_df["dogs"].nlargest(10)
    print(most_like_dogs)

    print('\n\nChange the last word of the first sentence from "pets" to '
          '"animals", and "humans" is no longer close to dogs.')
    np.random.seed(SEED)
    random.seed(SEED)
    animals_corpus = corpus.copy()
    animals_corpus[0][-1] = "animals"
    w_vectors, c_vectors, index2word, word2index = train_sgns(animals_corpus,
                                                              vector_size=10,
                                                              window=2,
                                                              negative=1,
                                                              epochs=20)
    similarity_df = pd.DataFrame([[cos_similarity(w_vectors[i], w_vectors[j])
                                   for j in range(len(w_vectors))]
                                  for i in range(len(w_vectors))],
                                 columns=index2word,
                                 index=index2word)
    most_like_dogs = similarity_df["dogs"].nlargest(10)
    print(most_like_dogs)

    # sns.heatmap(similarity_df)
    # plt.show()
