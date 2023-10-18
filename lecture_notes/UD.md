# Universal Dependencies v1

> Nivre, Joakim, Marie-Catherine De Marneffe, Filip Ginter, Yoav Goldberg, Jan
> Hajic, Christopher D. Manning, Ryan McDonald et al. "Universal dependencies
> v1: A multilingual treebank collection." In Proceedings of the Tenth
> International Conference on Language Resources and Evaluation (LREC'16), pp.
> 1659-1666. 2016.

## 3. Annotation guideline principles

> syntactic wordhood does not always coincide with whitespace-separated
> orthographic units

### 3.1. Word segmentation

***Syntactic*** words (words are split when syntactic analysis requires)

### 3.2. Morphology (how words change/"inflect")

Identify lemma/POS/morphosyntactic properties in Figure 2.

### 3.3. Syntax

> By design, UD indicates in the dependency labels whether dependents are
> phrases or clauses

#### compounding

* *mwe()*: de facto
* *name*: Hillary Rodham Clinton
* *compound()* put up

#### Relations between content words

> Each word depends either on another word in the sentence or on a notional
> “root” of the sentence, following three principles: content words are related
> by dependency relations; function words attach to the content word they
> further specify; and punctuation attaches to the head of the phrase or clause
> in which it appears, as illustrated in Figure 2.

Explain (1) and (2)

> To have relations between content words, any case-marking element (including
> prepositions, postpositions, and clitic case markers) is treated as a
> dependent of the noun it attaches to or introduces

Explain (3)

>  Coordination follows a similar treatment: the leftmost conjunct is the head,
>  and other conjuncts as well as the coordinating conjunction depend on it

Explain (4)

>  The UD view is that we need to recognize both lexical and functional heads,
>  but in order to maximize parallelism across languages, only lexical heads
>  are inferable from the topology of our tree structures.

#### Language-specific relations

Dependency labels can add subtypes after a colon

### 3.4. Format and [tools](https://universaldependencies.org/tools.html)

CoNLL-U format (see Figure 3)

1. unique id (integer for words, ranges for multiword tokens)
1. word form
1. lemma
1. universal part-of-speech tag
1. optional language-specific part-of-speech tag
1. morphological features
1. head
1. dependency relation
1. additional dependencies in the enhanced representation
1. miscellaneous information



## 4. [Existing treebanks](https://universaldependencies.org/)

37 treebanks (33 languages) (Now, 245 treebanks in 141 languages)

## 5. Conclusion

> we hope to be able to release tools for tokenization, morphological analysis
> and syntactic parsing for all languages

```console
$ python3 -m pip install stanza
```

See [stanza](https://stanfordnlp.github.io/stanza/).
Refer to [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)

```python
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
doc = nlp("Barack Obama was born in Hawaii.") # Run the pipeline on the input text
print(doc)

cu_nlp = stanza.Pipeline('cu')  # Old Church Slavonic
cu_doc = cu_nlp("остави тоу даръ твои прѣдъ олътаремъ ꙇ шедъ прѣжде съмири сѧ съ братромъ своимъ и тогда пришедъ принеси даръ твои")
print(f"{cu_doc:C}")  # :C format specifier for CoNLL-U output
```

# [hunpos](https://github.com/mivoq/hunpos) demo, if time permits
