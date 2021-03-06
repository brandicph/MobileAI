import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import logging
import sys
import os

from scipy import interpolate
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
import statsmodels
from matplotlib2tikz import save as tikz_save
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.metrics import confusion_matrix, classification_report

from pandas.plotting import andrews_curves

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold

from qp import QualiPoc
from svm import SVM, Kernels

mpl.style.use('seaborn')

def plot_contour(X1_train, X2_train, clf):
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    plt.axis("tight")
    plt.show()


# CSV file path
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
# S-TOG Ping (DTU)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# DTU 1GB download
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
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


#qp.df = qp.normalize(['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'Intermediate KPI'], overwrite=True)

# Perform filtering
n_samples_before_filtering = len(qp.df)
qp.logger.info("Before filtering: {} samples".format(n_samples_before_filtering))
df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

df = df[df['Intermediate KPI'] > 0]
#df = df[df['SINR Rx[0]'] > 0]
#df = df[df['RSRP Rx[0]'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))

# Perform rolling mean
rolling_window = 100
df_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'Intermediate KPI']]#.rolling(rolling_window).sum() / rolling_window

df_rolling = df_rolling.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

n_samples_after_rolling = len(df_rolling)
qp.logger.info("After rolling: {} samples".format(n_samples_after_rolling))


plot_data = df_rolling[['SINR Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'RSRP Rx[0]','Intermediate KPI']][::5]
plot_data = plot_data[plot_data['Intermediate KPI'] > 20000]

#sns.pairplot(plot_data, diag_kind="kde", markers="+", plot_kws=dict(s=50, edgecolor="#C44E52", linewidth=1), diag_kws=dict(shade=True))#, dropna=True)
#plt.show(block=False)

cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])


pgf_with_custom_preamble = {
    "font.family": 'serif',
    "font.serif": 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
    "font.sans-serif": 'Helvetica, Avant Garde, Computer Modern Sans serif',
    "font.cursive": 'Zapf Chancery',
    "font.monospace": 'Courier, Computer Modern Typewriter',
    "text.usetex": True,
    "text.dvipnghack": True
}
mpl.rcParams.update(pgf_with_custom_preamble)


plot_data.rename(columns={
    'RSRP Rx[0]': r'RSRP $Rx_0$',
    'RSRP Rx[1]': r'RSRP $Rx_1$',
    'SINR Rx[0]': r'SINR $Rx_0$',
    'SINR Rx[1]': r'SINR $Rx_1$',
}, inplace=True)

sns.set_context("notebook", rc={"lines.linewidth": 1})
g = sns.PairGrid(plot_data)
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
#sns.pairplot(plot_data, kind='reg', diag_kind="kde", markers="+", diag_kws=dict(shade=True, ))#, dropna=True)
plt.show(block=False)

g.savefig("pairplot_sinr_rsrp_kpi.pdf")

g = sns.PairGrid(plot_data[[r'SINR $Rx_0$', r'RSRP $Rx_0$', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
#sns.pairplot(plot_data, kind='reg', diag_kind="kde", markers="+", diag_kws=dict(shade=True, ))#, dropna=True)
plt.show(block=False)
g.savefig("pairplot_sinr0_rsrp0_kpi.pdf")

jointplot001 = sns.jointplot(x=r'RSRP $Rx_0$', y="Intermediate KPI", data=plot_data, x_estimator=np.mean, kind="reg", color='#C44E52', size=7, scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
plt.show(block=False)
jointplot001.savefig("jointplot_rsrp0_kpi.pdf")

jointplot002 = sns.jointplot(x=r'RSRP $Rx_1$', y="Intermediate KPI", data=plot_data, x_estimator=np.mean, kind="reg", color='#C44E52', size=7, scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
plt.show(block=True)
jointplot002.savefig("jointplot_rsrp1_kpi.pdf")


"""
plot_data2 = df_rolling[['SINR Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'RSRP Rx[0]','Intermediate KPI']][::5]
plot_data2 = plot_data2[plot_data2['Intermediate KPI'] < 40000]

sns.jointplot(x="RSRP Rx[0]", y="Intermediate KPI", data=plot_data2, kind="reg", color='#4C72B0', size=7, line_kws=dict(lw=2, color="black"))

plt.show()
"""

"""
msk = np.random.rand(len(df_rolling)) < 0.8

train = df_rolling[msk]
X_train = train[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]']]
y_train = train[['Intermediate KPI']]

test = df_rolling[~msk]
X_test = test[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]']]
y_test = test[['Intermediate KPI']]
"""

"""
train = df_rolling.sample(frac=0.8,random_state=200)
test = df_rolling.drop(train.index)

X_train = train[['SINR Rx[0]', 'RSRP Rx[0]']].values
y_train = train[['Intermediate KPI']].values

X_test = test[['SINR Rx[0]', 'RSRP Rx[0]']].values
y_test = test[['Intermediate KPI']].values

clf = SVM(kernel=Kernels.Gaussian)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
correct = np.sum(y_predict == y_test)

#cr = classification_report(y_test, y_predict)
#cm = confusion_matrix(y_test, y_predict)

#print("\n\nClassification report for classifier %s:\n%s\n" % (clf, cr))

#print("Confusion matrix:\n%s" % cm)

print("%d out of %d predictions correct" % (correct, len(y_predict)))
"""

#clf = SVM(kernel=Kernels.Gaussian)
#scores = cross_val_score(clf, df_rolling[['SINR Rx[0]', 'RSRP Rx[0]']].values, df_rolling[['Intermediate KPI']].values, cv=6)
#print("Cross-validated scores:", scores)

"""
X = df_rolling[['SINR Rx[0]', 'RSRP Rx[0]']].values[100:1000]
y = df_rolling[['Intermediate KPI']].values[100:1000]

kf = KFold(n_splits=6) # Define the split - into 2 folds 

print(kf)

for train_index, test_index in kf.split(X):
	#print("TRAIN:", train_index, "TEST:", test_index)
	print("New fold")
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]



	clf = SVM(kernel=Kernels.Gaussian)
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_test)
	correct = np.sum(y_predict == y_test)

	print("%d out of %d predictions correct" % (correct, len(y_predict)))
"""

"""
classes = ["1","0"]
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
plt.figure()
sns.heatmap(df_cm, annot=True, cmap="Reds")
plt.show(block=False)
"""

#plt.figure()
#plot_contour(X_train[y_train > 0.5], X_train[y_train < 0.5], clf)


#test_specific(kernel=Kernels.Linear, data=gen_lin_separable_data)
#test_specific(kernel=Kernels.Linear, data=gen_lin_separable_data, C=1.0)
#test_specific(kernel=Kernels.Polynomial, data=gen_non_lin_separable_data)
#test_specific(kernel=Kernels.Gaussian, data=gen_non_lin_separable_data)
#test_specific(kernel=Kernels.Sigmoid, data=gen_discrete_separable_data)
#test_specific(kernel=Kernels.RationalQuadratic, data=gen_non_lin_separable_data)
#test_specific(kernel=Kernels.MultiQuadric, data=gen_non_lin_separable_data)
#test_specific(kernel=Kernels.InverseMultiQuadric, data=gen_non_lin_separable_data)