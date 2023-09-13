
> Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, and Jeffrey Dean.
> 2007. Large Language Models in Machine Translation. In Proceedings of the
> 2007 Joint Conference on Empirical Methods in Natural Language Processing and
> Computational Natural Language Learning (EMNLP-CoNLL), pages 858–867, Prague,
> Czech Republic. Association for Computational Linguistics.

# Abstract

Sounds like this article is about computational efficiency without sacrificing
results.

# 1 Introduction

1. How might one build a language model that allows scaling to very large amounts of training data?
   - One possible answer: stupid backoff
2. How much does translation performance improve as the size of the language model increases?
   - Learning curves for one MT system
3. Is there a point of diminishing returns in performance as a function of language model size?
   - TBD

# 2 N-gram language models

Describes n-grams, sparse data problem, and smoothing

# 3 Related work on distributed language models

Previous methods require querying multiple workers to get n-gram counts. Our
method requires one (or two).

# 4 Stupid Backoff

Previous context-dependent backoff uses lambda weights to generate normalized
probabilities.

Stupid backoff produces "scores", not probabilities.

> The lack of normalization in Eq. (5) does not affect the functioning of the
> language model in the present setting, as Eq. (1) depends on relative rather
> than absolute feature-function values.

# 5 Distributed training

Map-reduce is used to generate token ids (dictionary), sharding the same tokens
to the same nodes for counting.

Map-reduce is also used to generate n-grams (of token ids). Sharding is based
on hash of the first two tokens. Each shard contains unigram counts of its
initial tokens, so everything needed for stupid backoff is contained in each
shard.

KN smoothing and Katz backoff require MUCH more compute.

# 6 Distributed application

Decoder queues requests in batches to minimize effect of network latency.

It extends each hypothesis by one token, queues needed n-grams, requests
n-grams, scores hypotheses, prunes, then extends hypotheses by one token, etc.

"Beam search" example uses beam size of 4.

On average 150,000 n-grams requested per sentence.

# 7 Experiments




# Main takeaways

1. Scaling language models is HARD.
2. Sometimes abandoning mathematical/statistical idealism for computational
   efficiency is justified.
   - The lack of normalization in Eq. (5) does not affect the functioning of
     the language model in the present setting, as Eq. (1) depends on relative
     rather than absolute feature-function values.  
3. Write your article like a textbook. Introduce models, metrics, etc. as if
   your reader has never heard of them.
4. There's no data like more data.  (GPT-4 was trained on ~13T tokens)

# Criticisms

Leaving out KN-smoothing for `web` was understandable, but lame. We have to
imagine that the BLEU trend continues with the largest size of LMs.

>  Perplexities for ldcnews range from 351.97 to 210.93 and are also close to
>  linear (r^2 = 0.987), while those for webnews data range from 221.85 to
>  164.15 and flatten out near the end.

"Flatten out near the end."


