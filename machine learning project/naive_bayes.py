import numpy as np

class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        sample_num = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.log_prior = [np.log(len(i) / sample_num) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob * x).sum(axis=1) + self.log_prior
                for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
    
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
