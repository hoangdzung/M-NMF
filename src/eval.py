from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import numpy as np 
import pandas as pd
import sys 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv")
parser.add_argument("--x")
parser.add_argument("--y")
parser.add_argument("--prob")
parser.add_argument('--kmean',action="store_true")
args = parser.parse_args()

y = np.load(args.y)
if args.kmean:
    X1 = np.load(args.x)
    n_labels = max(y) +1 
    y_kmeans = KMeans(n_clusters=n_labels, random_state=0).fit_predict(X1)
    X2 = np.zeros((X1.shapem, n_labels))
    X2[np.arange(X1.shape[0], n_labels)] = 1
    X = np.hstack([X1, X2])
else:
    if args.csv is not None:
        df = pd.read_csv(args.csv, sep=',')
        X = df.values
    else:
        X1 = np.load(args.x)
        df = pd.read_csv(args.prob, sep=',')
        X2 = df.values
        X = np.hstack([X1, X2])

scores = []
for seed in tqdm(range(50)):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed)
    if y_train.ndim == 1:
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                multi_class='ovr',max_iter=4000).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    else:
        clf = TopKRanker(LogisticRegression(random_state=0, solver='lbfgs',
                multi_class='ovr',max_iter=4000)).fit(X_train, y_train)
            score = clf.score(X_test, y_test.sum(1).astype(int), y_test)
    scores.append(score)
scores = np.array(scores)
print("{},{}".format(np.mean(scores), np.std(scores)))