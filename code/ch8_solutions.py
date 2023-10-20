from collections import Counter
from collections import defaultdict

import nltk
from nltk.corpus import brown
import numpy as np

nltk.download("brown")
# nltk.download("universal_tagset")  # TODO find way to limit to Penn Tagset

brown_list = list(brown.tagged_words())
train_set = brown_list[:928944]
dev_set = brown_list[928944:]

print('all tags:', Counter(tag for tok, tag in train_set))

### Exercise 8.4 ###


def build_most_likely_tag_model(corpus):
    continuations = defaultdict(Counter)
    for token, tag in corpus:
        continuations[token].update([tag])
    most_likely_tags = {token: max(tag_counts, key=tag_counts.get)
                        for token, tag_counts in continuations.items()}
    # You can use tag_counts.most_common(1)[0][0] to do the same thing:
    # most_likely_tags = {token: tag_counts.most_common(1)[0][0]
    #                     for token, tag_counts in continuations.items()}
    return most_likely_tags


def backoff_rules(tok):
    # TODO make decisions
    return 'NN'


most_likely_tags = build_most_likely_tag_model(train_set)
most_likely_dev = [(tok, most_likely_tags.get(tok, backoff_rules(tok)))
                   for tok, true_tag in dev_set]
most_likely_acc = sum(true_tag == pred_tag
                      for (_, true_tag), (_, pred_tag) in zip(dev_set, most_likely_dev)) / len(dev_set)
print(f"{most_likely_acc=}")


### Exercise 8.5 ###


def viterbi(obs, states, initial_prob, trans_prob, emit_prob):
    """The Viterbi algorithm for finding the most likely sequence of POS tags"""
    T = len(obs)
    N = len(states)

    # Initialize the Viterbi matrix and backpointer matrix
    viterbi_mat = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialize the first column of the Viterbi matrix
    for i in range(N):
        viterbi_mat[i, 0] = initial_prob[states[i]] * emit_prob[states[i]][obs[0]]
        backpointer[i, 0] = -1

    # Fill in the Viterbi matrix
    for t in range(1, T):
        for s in range(N):
            max_prob = 0
            max_state = 0
            for s_prime in range(N):
                prob = viterbi_mat[s_prime, t - 1] * trans_prob[states[s_prime]][states[s]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = s_prime
            viterbi_mat[s, t] = max_prob * emit_prob[states[s]][obs[t]]
            backpointer[s, t] = max_state

    # Find the best path by backtracking
    best_path = [0] * T
    best_last_state = np.argmax(viterbi_mat[:, -1])
    best_path[-1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    return [states[s] for s in best_path]


### define HMM model ###

# Define the set of states (POS tags) and the set of observations (words)
states = ['NNS', 'VBZ', 'JJ']
observations = ['dog', 'barks', 'loud']

# Define the initial probabilities
initial_probabilities = {'NNS': 0.4, 'VBZ': 0.3, 'JJ': 0.3}

# Define the transition probabilities (see 8.8)
transition_probabilities = {
    'NNS': {'NNS': 0.2, 'VBZ': 0.7, 'JJ': 0.1},
    'VBZ': {'NNS': 0.3, 'VBZ': 0.4, 'JJ': 0.3},
    'JJ': {'NNS': 0.1, 'VBZ': 0.6, 'JJ': 0.3},
}

# Define the emission probabilities (see 8.10)
emission_probabilities = {
    'NNS': {'dog': 0.8, 'barks': 0.1, 'loud': 0.1},
    'VBZ': {'dog': 0.1, 'barks': 0.7, 'loud': 0.2},
    'JJ': {'dog': 0.2, 'barks': 0.1, 'loud': 0.7},
}


# Test the HMM with an example sentence
sentence = ['dog', 'barks', 'loud']
best_path = viterbi(sentence, states, initial_probabilities,
                    transition_probabilities, emission_probabilities)
print("Input Sentence:", sentence)
print("Predicted POS Tags:", best_path)
