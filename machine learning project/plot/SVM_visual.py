import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm

data = pd.read_csv('../data/training.csv')

X = data[['recent_ave_rank', 'jockey_ave_rank']].values
y = data[['HorseRankTop50Percent']].values.ravel()

svm_model = svm.SVC(C=1, kernel='linear', random_state=3320)
svm_model.fit(X, y)

h = 0.02 # step size in the mesh

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, alpha=0.7)
plt.xlabel('Recent average rank', size=11)
plt.ylabel('Jockey average rank', size=11)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVC with linear kernel', size=15)
plt.legend()

plt.savefig('SVM_visual.png')
plt.show()