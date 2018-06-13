import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv('../data/training.csv')

labels = ['recent_ave_rank', 'draw', 'race_distance', 'trainer_ave_rank', 'jockey_ave_rank', 'actual_weight', 'declared_horse_weight', 'win_odds']
X = data[['recent_ave_rank', 'draw', 'race_distance', 'trainer_ave_rank', 'jockey_ave_rank', 'actual_weight', 'declared_horse_weight', 'win_odds']].values
y = data[['HorseWin']].values.ravel()

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

x_labels = []
for idx in indices:
    x_labels.append(labels[idx])

plt.figure(figsize=(15, 15))
plt.title("Feature importances", size=15)
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center", width=0.5)

plt.xticks(range(X.shape[1]), x_labels)
plt.xlim([-1, X.shape[1]])
plt.xlabel('Features', size=11)
plt.ylabel('Horse Win', size=11)
plt.savefig('bar_chart.png')
plt.show()