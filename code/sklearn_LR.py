"""
Logistic regression using sci-kit learn.
Some code adapted from...
    * https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
"""

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups  # toy text classification data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(verbose=True, remove=()):
    """Load and vectorize the 20 newsgroups dataset."""

    data_train = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    # order of labels in `target_names` can be different from `categories`
    target_names = data_train.target_names

    # split target in a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    # Extracting features from the training data using a sparse vectorizer
    # NOTE look up what max_df, min_df, and stop_words do
    vectorizer = CountVectorizer(max_df=0.5, min_df=5, stop_words="english")
    X_train = vectorizer.fit_transform(data_train.data)

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(data_test.data)

    feature_names = vectorizer.get_feature_names_out()

    if verbose:
        print(f"{len(data_train.data)} documents - (training set)")
        print(f"{len(data_test.data)} documents - (test set)")
        print(f"{len(target_names)} categories")
        print(f"Train n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(f"Test n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names, target_names


X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset()
print("X_test (sparse matrix format): ", X_test)
print("feature_names: ", feature_names)
print("target_names: ", target_names)

clf = LogisticRegression(penalty="l2")
clf.fit(X_train, y_train)

pprint(clf.__dict__)

print("Evaluating features...")
best_features = {}
for i, class_weights in enumerate(clf.coef_):
    sorted_weights = sorted(zip(feature_names, class_weights),
                            key=lambda x: x[1], reverse=True)  # key sorts by class_weights
    best_features[target_names[i]] = sorted_weights[:10]
pprint(best_features)

y_test_hat_probabilities = clf.predict_proba(X_test)
print("probabilities: ", y_test_hat_probabilities)
y_test_hat = clf.predict(X_test)
print("predicted classes: ", y_test_hat)
y_test_hat_names = np.array([target_names[i] for i in y_test_hat])
print("predicted class names: ", y_test_hat_names)
score = clf.score(X_test, y_test)
print("Mean accuracy: ", score)

report = classification_report(y_test, y_test_hat, target_names=target_names)  # output_dict=True if you want as a data structure
print(report)

ConfusionMatrixDisplay.from_predictions(y_test, y_test_hat, display_labels=target_names)
plt.show()
