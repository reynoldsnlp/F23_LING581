# DistilBERT

> Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT,
> a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv
> preprint arXiv:1910.01108 (2019).

# 1 Intro

Problems with the rise of bigger models

* environment cost
* adoption

40% smaller, 60% faster, similar performance

# 2 Knowledge distillation

Student trained on the soft target probabilities of the teacher

Q: What is softmax-temperature?

#3 DistilBERT

Architecture

* remove token-type embeddings
* remove pooler
* reduce number of layers (linear layer and layer norm)

Initialization

* take one out of BERT's two layers

Distillation

* very large batches
* no next sentence prediction objective

Compute

* 8 16GB V100 GPUs for 90 hours

# 4 Experiments

Outperforms BERT on WNLI:

* A Winograd schema is a pair of sentences that differ in only one or two words
  and that contain an ambiguity that is resolved in opposite ways in the two
  sentences and requires the use of world knowledge and reasoning for its
  resolution. The schema takes its name from a well-known example by Terry
  Winograd: *The city councilmen refused the demonstrators a permit because they
  [feared/advocated] violence.*
