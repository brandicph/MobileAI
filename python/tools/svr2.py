import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import logging
import sys
import os

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import inspect
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV

from qp import QualiPoc
mpl.style.use('seaborn')

"""
http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html
"""

# #############################################################################
# Generate sample data
#X = np.sort(5 * np.random.rand(40, 1), axis=0)
#y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))


# #
# Read data
# CSV file path
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
# S-TOG Ping (DTU)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_tracking.csv')
# S-TOG Ping (Bx)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-04-08-41-51-0000-5310-7746-0004-S_ping2.csv')
# Airport Signal Quality
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_signal_quality.csv')
# Airport Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_airport_ping2.csv')
# City Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-17-11-58-0000-5310-7746-0004-S_city_ping2.csv')

# Instatiate data reader
qp = QualiPoc(CSV_IN_FILEPATH)

# Perform filtering
n_samples_before_filtering = len(qp.df)
qp.logger.info("Before filtering: {} samples".format(n_samples_before_filtering))
#fields = ['Intermediate KPI', 'RSRP Rx[0]']
#X_fields = ['RSRP Rx[0]']
#fields = ['Intermediate KPI', 'RSRP', 'RSRQ', 'RSSI', 'Average MCS Index']
#X_fields = ['RSRP','RSRQ', 'RSSI', 'Average MCS Index']
#y_fields = ['Intermediate KPI']
fields = ['Intermediate KPI', 'RSRP']
X_fields = ['RSRP']
y_fields = ['Intermediate KPI']
#qp.df = qp.normalize(fields, overwrite=True)
#df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])
df = qp.df.dropna(subset=fields)

df = df[df['Intermediate KPI'] > 0]
#df = df[df['SINR Rx[0]'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))

#df = df[0:500]
#df = df[::5]

# Perform rolling mean
rolling_window = 100
#df_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'Intermediate KPI']].rolling(rolling_window).sum() / rolling_window
#df_rolling = df_rolling.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

#df = df.groupby('Time').agg({"Intermediate KPI": np.mean, "RSRP": lambda x: x.nunique()})
df_rolling = df[fields].rolling(rolling_window).sum() / rolling_window
df_rolling = df_rolling.dropna(subset=fields)
df_rolling = df_rolling.sort_values(by=X_fields)
df_rolling = df_rolling[::5]


n_samples_after_rolling = len(df_rolling)
qp.logger.info("After rolling: {} samples".format(n_samples_after_rolling))


X = df_rolling[X_fields].values
y = df_rolling[y_fields].values.ravel()

# ############################################################################
# KFold

kf = KFold(n_splits=2,shuffle=True) # Define the split - into 2 folds 

print(kf)

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index[::]]
    y_train, y_test = y[train_index], y[test_index[::]]
    print("New fold: ", X_train.shape)

    clf = SVR(kernel='rbf', C=1e3, gamma=0.01)
    #clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
    clf = clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    errors = np.abs(y_predict-y_test)/np.abs(y_test)
    error_avg = np.mean(errors)
    error_max = np.max(errors)
    error_min = np.min(errors)
    #mse = np.mean((y_test - y_predict)**2)
    print('Mean error', error_avg)
    print('Min error', error_min)
    print('Max error', error_max)
    #print('Mean square error', mse)
    correct = np.sum(errors <= 0.05)

    print("%d out of %d (%.4f) predictions correct" % (correct, len(y_predict), correct/len(y_predict)))

    plt.scatter(X_train, y_train, color='darkorange', label='data')
    plt.plot(X_train, clf.predict(X_train), color='navy', lw=2, label='RBF train')
    plt.scatter(X_test, y_test, color='green', marker='+', label='RBF test')
    plt.scatter(X_test, y_predict, color='purple', marker='+', label='RBF predict')
    #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    quit()


# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
