# Two feature weighting approaches for naive Bayes text classifiers

> Lungan Zhang, Liangxiao Jiang, Chaoqun Li, Ganggang Kong,
> Two feature weighting approaches for naive Bayes text classifiers,
> Knowledge-Based Systems,
> Volume 100,
> 2016,
> Pages 137-144,
> ISSN 0950-7051,
> https://doi.org/10.1016/j.knosys.2016.02.017.
> (https://www.sciencedirect.com/science/article/pii/S0950705116001039)

# Abstract

Two feature-weighting approaches for text classification that both show
improved accuracy, and maintain simplicity and fast execution time.

# 1 Introduction

Previous NB approaches to text classification:

* multi-variate Bernoulli naive Bayes (BNB)
  * Each word is a boolean
* multinomial naive Bayes (MNB)
  * Word frequencies
  * Unbalanced training corpus -> poor weights for decision boundary
* complement naive Bayes (CNB) 
* One-versus-all-but-one (OVA)
  * Combination of MNB and CNB

## Feature weighting

* Chi-squared-based runs fast but doesn't improve much
* correlation-based feature selection (CFS) improved accuracy, but runs slow

Adapting NB feature-weighting approaches to text classification

# 2 Related work

Formulae for MNB, CNB, OVA. (1-7)

Feature weighting formulae (10-12)

How to learn the feature weights?

* chi-squared measures positive term-class dependency
  * applied to BNB only, so not better performance than MNB, CNB, OVA
* correlation-based feature selection (CFS) selects a best feature subset
  * Quadratic time complexity
* Gain ratio-based feature weighting based on information gain
  * Reduction in entropy
* Decision tree-based feature gives lower weights to features that have many dependencies
* differential evolution algorithm-based feature weighting 

# 3 Two adaptive feature weighting approaches

## 3.1 Gain ratio-based feature weighting

define the gain ratio of each feature (word) partitioning a collection of training documents in (18).

Incorporate the learned feature weights into both the classification formulas and their conditional probability estimates.

## 3.2 Decision tree-based feature weighting

See formula (21).
Lambda can be any integer 3-10 (and higher?).
Incorporate the learned feature weights into both the classification formulas and their conditional probability estimates.

# 4 Experiments and results

Applying feature weighting approaches to 3 main algorithms on many corpora is
almost always very advantageous.

Training time is better than CFS.

Real-world task with customer reviews (is this really necessary?)

## Discussion

Can be used for other algorithms as well, such as KNN 


# Criticisms

* Define acronyms before you use them.
  * "CFS-based feature weighting algorithm"
* So... many... tables.
* Probably trying to do too much? (The bit about KNNs should be a different article) 
