from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import numpy as np 
import pandas as pd
import sys 

xcsvfile = sys.argv[1]
ynpyfile = sys.argv[2]

df=pd.read_csv(xcsvfile, sep=',')
X = df.to_numpy()
y = np.load(ynpyfile)

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