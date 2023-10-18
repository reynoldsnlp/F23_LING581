# How to access the Brown corpus for training data

import nltk
from nltk.corpus import brown
nltk.download("brown")

brown_list = list(brown.tagged_words())
train_set = brown_list[:928944]
dev_set = brown_list[928944:]
