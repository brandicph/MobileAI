import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from sklearn.model_selection import KFold


# #############################################################################
# Generate sample data

def gen_non_lin_separable_data():
    mean1 = [1, 1]
    mean2 = [-1, -1]
    mean3 = [-4, -4]
    mean4 = [4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.random.rand(len(X1)) + 1#np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.random.rand(len(X2)) - 1 #np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 10])
    mean2 = np.array([10, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1)) * +1#np.random.rand(len(X1)) + 1
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1#np.random.rand(len(X2)) - 1
    return X1, y1, X2, y2


X1, y1, X2, y2 = gen_lin_separable_data()
y = np.hstack((y1,y2))
X = np.vstack((X1,X2))

print(len(X),len(y))

kf = KFold(n_splits=2)


# #############################################################################
# Add noise to targets
#y[::5] += 10 * np.random.rand()

# #############################################################################
# Fit regression model


train_sizes, train_scores, test_scores = learning_curve(SVR(kernel='rbf',C=1.0, epsilon=0.2), X, y, cv=4)

plt.figure()
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show(block=False)

plt.figure()
plt.plot(X,y,'+')
plt.show()
"""
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svr = SVR(kernel='linear', C=1e3, gamma=0.1)
    model = svr.fit(X_train, y_train)

    y_result = model.predict(X_test)

    print("y_test:", y_test, "y_result:", y_result)
"""
"""
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
"""
# #############################################################################
# Look at the results
"""
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
"""