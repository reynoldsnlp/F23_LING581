"""Exercises 3.8 - 3.11 from Speech and Language Processing v3."""

from collections import Counter
from collections import defaultdict
from pprint import pprint
from random import choice
from random import choices


text1 = ['<s>', 'this', 'is', 'my', 'example', 'text', '.', '</s>', '<s>', 'this', 'is', 'not', 'my',
         'example', 'test', '.', '</s>', '<s>', 'that', "'s", 'my', 'point', '.', '</s>']


# 3.8: Write a program to compute unsmoothed unigrams and bigrams.

def freqs_from_zip_slices(text, n):
    """Compute n-gram frequencies using zip().

    Parameters
    ----------
    text : List[str]
        Tokenized list of strings.
    n : int
        Order of the n-grams (unigram=1, bigram=2, trigram=3, 4-gram=4, etc.)
    """
    ngram_freqs = Counter(zip(*[text[i:] for i in range(n)]))
    return ngram_freqs


def freqs_from_zip_iters(text, n):
    """Compute n-gram frequencies using zip() of iterators,
    rather than of copied slices (more memory efficient).

    Parameters
    ----------
    text : List[str]
        Tokenized list of strings.
    n : int
        Order of the n-grams (unigram=1, bigram=2, trigram=3, 4-gram=4, etc.)
    """
    iters = []
    for i in range(n):
        this_iter = iter(text)
        for j in range(i):
            next(this_iter)
        iters.append(this_iter)
    ngram_freqs = Counter(zip(*iters))
    return ngram_freqs


def freqs_from_indices(text, n):
    """Compute n-gram frequencies using indices.

    Parameters
    ----------
    text : List[str]
        Tokenized list of strings.
    n : int
        Order of the n-grams (unigram=1, bigram=2, trigram=3, 4-gram=4, etc.)
    """
    ngram_freqs = Counter()
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i:i + n])
        ngram_freqs.update([ngram])
    return ngram_freqs


def compute_ngram_model(freqs, smoothing=None):
    """Compute n-gram model from frequencies.

    Parameters
    ----------
    freqs : Dict[Tuple[str], float]
        N-gram frequency distribution with tuples of tokens as keys
        and (weighted?) frequencies as values
    smoothing : Optional[str], default=None
        Smoothing algorithm to apply
    """
    cont_dict = defaultdict(dict)
    ngram_model = defaultdict(dict)
    # The following loop could have been achieved while compiling the freqs
    for ngram, count in freqs.items():
        *prefix, continuation = ngram
        prefix = tuple(prefix)  # lists are not hashable so cannot be keys
        cont_dict[prefix][continuation] = count
    if smoothing is None:
        for prefix, cont_dist in cont_dict.items():
            N = sum(cont_dist.values())
            p_dist = {c: n / N for c, n in cont_dist.items()}
            ngram_model[prefix] = p_dist
    else:
        raise NotImplementedError('The requested smoothing technique is not '
                                  'implemented.')
    return ngram_model


if __name__ == '__main__':
    for n in range(1, 3):
        from_slices = freqs_from_zip_slices(text1, n)
        from_iters = freqs_from_zip_iters(text1, n)
        from_indices = freqs_from_indices(text1, n)
        assert from_slices == from_iters == from_indices, [from_slices, from_iters, from_indices]

    ngram_model = compute_ngram_model(from_slices)
    pprint(ngram_model)


# 3.9: Run your n-gram program on two different small corpora of your choice
# (you might use email text or newsgroups). Now compare the statistics of the
# two corpora. What are the differences in the most common unigrams between the
# two? How about interesting differences in bigrams?


# 3.10: Add an option to your program to generate random sentences.

def generate_sentence(ngram_model, start=None, weighted=False, stop_tok='</s>',
                      max_len=100):
    """Generate a random sentence using the given n-gram model.

    Parameters
    ----------
    ngram_model : Dict[Tuple[str], Dict[str, float]]
        N-gram model with "given" n-gram as keys, and the probability
        distribution of continuation strings as values. The latter is itself a
        dictionary with strings as keys and floats as values.
    start : Optional[Iterable[str]], default=None
        Tokens used to start the sentence. Must be the same length as the keys
        in `ngram_model`. If none is given, start will be randomly generated
        from `ngram_model.keys()`.
    weighted : bool, default=False
        Whether to weight the random selection according to model frequencies
    stop_tok : str, default='</s>'
        Token after which to return the sentence
    max_len : int, default=100
        Maximum sentence length. If the generated sentence reaches this length,
        it is returned without reaching the terminal `stop_tok`.
    """
    order = len(next(iter(ngram_model.keys())))  # prefix n-gram size
    sent = []
    if start is None:
        start = choice(list(ngram_model.keys()))
    sent.extend(start)
    if weighted:
        while sent[-1] != stop_tok and len(sent) < max_len:
            cont_dist = ngram_model[tuple(sent[-order:])]
            next_tok = choices(list(cont_dist.keys()),
                               weights=list(cont_dist.values()))
            sent.extend(next_tok)
    else:
        while sent[-1] != stop_tok and len(sent) < max_len:
            cont_dist = ngram_model[tuple(sent[-order:])]
            next_tok = choice(list(cont_dist.keys()))
            sent.append(next_tok)
    return sent


if __name__ == '__main__':
    print('Unweighted:')
    for _ in range(20):
        print(' '.join(generate_sentence(ngram_model)))
    print('\nWeighted:')
    for _ in range(20):
        print(' '.join(generate_sentence(ngram_model, weighted=True)))


# 3.11: Add an option to your program to compute the perplexity of a test set. 


def perplexity(text, ngram_model):
    """Compute perplexity of `text` based `ngram_model`.

    Parameters
    ----------

    text : List[str]
        A tokenized text.
    ngram_model : Dict[Tuple[str], Dict[str, float]]
        N-gram model with "given" n-gram as keys, and the probability
        distribution of continuation strings as values. The latter is itself a
        dictionary with strings as keys and floats as values.
    """
    pass 

