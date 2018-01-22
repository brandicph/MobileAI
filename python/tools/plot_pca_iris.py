#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
PCA example with Iris Data-set
=========================================================

Principal Component Analysis applied to the Iris dataset.

See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import csv
from datetime import datetime,timedelta
import logging
import sys
import os

import pandas as pd
from pandas.plotting import andrews_curves

import numpy as np

from scipy import interpolate
from scipy.stats import kendalltau

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn.datasets import fetch_mldata

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

mpl.style.use('seaborn')

# pomegranate: #c0392b 
pgf_with_custom_preamble = {
    "font.family": 'serif',
    "font.serif": 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
    "font.sans-serif": 'Helvetica, Avant Garde, Computer Modern Sans serif',
    "font.cursive": 'Zapf Chancery',
    "font.monospace": 'Courier, Computer Modern Typewriter',
    "text.usetex": True,
    "text.dvipnghack": True,
    #"figure.facecolor": 'white',
    "axes.facecolor": '#F5F5F5',
    "axes.color_cycle": ['#c0392b', '#7f8c8d', '#2c3e50', '#8e44ad', '#16a085']
}
mpl.rcParams.update(pgf_with_custom_preamble)


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
DATA_PATH = os.path.join(SCRIPT_PATH, '../../data/')
KPI_PATH = os.path.join(DATA_PATH, 'kpi/')
DATASET_PATH = os.path.join(DATA_PATH, 'signal_quality/')

def load_dataset(type='highway', n=5, merge_index=True):
    # Generate filepaths
    kpi_file_paths = [os.path.join(KPI_PATH, '{}/{}.csv'.format(type,x)) for x in np.arange(1,n+1)]
    dataset_file_paths = [os.path.join(DATASET_PATH, '{}/{}.csv'.format(type,x)) for x in np.arange(1,n+1)]

    # Load files into pandas
    kpi_df = [pd.read_csv(kpi_file_path) for kpi_file_path in kpi_file_paths]
    dataset_df = [pd.read_csv(dataset_file_path) for dataset_file_path in dataset_file_paths]

    # Set index for kpi
    for df in kpi_df:
        df.set_index('Time', inplace=True)

    # Set index for datasets
    for df in dataset_df:
        df.set_index('Time', inplace=True)

    # Determine to merge index
    how_to_merge = 'inner' if merge_index else 'outer'

    # Merge by kpi index
    results = []
    for i in np.arange(0,n):
        result = pd.merge(kpi_df[i], dataset_df[i], left_index=True, right_index=True, how=how_to_merge)
        result.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)
        result.dropna(subset=['Intermediate KPI'], inplace=True)
        results.append(result)

    return results

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')

datasets1_combined = pd.concat(datasets1)
#datasets1_combined['Type'] = 0
datasets2_combined = pd.concat(datasets2)
#datasets2_combined['Type'] = 1
datasets3_combined = pd.concat(datasets3)
#datasets3_combined['Type'] = 2

datasets123 = [datasets1_combined, datasets2_combined, datasets3_combined]
datasets123_combined = pd.concat(datasets123)
datasets123_combined['Type'] = 0
datasets123_combined.dropna(subset=['RSRP', 'RSRP Rx[0]', 'RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRQ', 'Average MCS Index', 'Intermediate KPI'],inplace=True)


#datasets123_combined = datasets123_combined.assign(Type = lambda x: 1)

#datasets123_combined['Type'] = np.where(datasets123_combined['Intermediate KPI'] < 75000 , 1, 2)
#datasets123_combined['Type'] = np.where(datasets123_combined['Intermediate KPI'] < 50000, 0, 1)

datasets123_combined.loc[(datasets123_combined['Intermediate KPI'] >= 60000), 'Type'] = 2
datasets123_combined.loc[(datasets123_combined['Intermediate KPI'] < 60000) & (datasets123_combined['Intermediate KPI'] >= 30000), 'Type'] = 1
datasets123_combined.loc[(datasets123_combined['Intermediate KPI'] < 30000), 'Type'] = 0

r = datasets123_combined['Type']

datasets123_combined = datasets123_combined[['RSRP', 'RSRP Rx[0]', 'RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRQ', 'Intermediate KPI']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
#X = datasets123_combined[['RSRP Rx[0]', 'RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]']]
#X = datasets123_combined[['RSRP Rx[0]','RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRQ']]
X = datasets123_combined[['RSRP', 'SINR Rx[0]', 'RSSI', 'RSRQ', 'Average MCS Index']]
#X = datasets123_combined[['RSRP Rx[0]','RSRP Rx[1]']]
y = datasets123_combined['Intermediate KPI']

#target_names = ['Highway', 'City', 'Country']
target_names1 = [r'Low ($bitrate < 30Mbps$)', r'Medium ($30Mbps <= bitrate < 60Mbps$)', r'High ($bitrate >= 60Mbps$)']
target_names2 = ['Low', 'Medium', 'High']

n_features = 2
U, _, _ = np.linalg.svd(X)
rank = np.linalg.matrix_rank(X)
pca = decomposition.PCA(n_components=n_features)
X_r = pca.fit(X).transform(X)

print('2 components', pca.components_)

n_components = np.arange(0, n_features, 5)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['#c0392b', '#7f8c8d', '#2c3e50']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names1):
    plt.scatter(X_r[r == i, 0], X_r[r == i, 1], color=color, alpha=.8, lw=1,
                label=target_name, edgecolor='k')
plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.tight_layout()
plt.savefig('simple_all_pca_high_medium_low_2d.pdf')
plt.show(block=False)


fig = plt.figure(1, figsize=(4, 3))
plt.clf()

n_features = 3
U, _, _ = np.linalg.svd(X)
rank = np.linalg.matrix_rank(X)
pca = decomposition.PCA(n_components=n_features)
X_r = pca.fit(X).transform(X)

n_components = np.arange(0, n_features, 5)

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# Reorder the labels to have colors matching the cluster results
for color, i, target_name1, target_name2 in zip(colors, [0, 1, 2], target_names1, target_names2):
    ax.text3D(X_r[r == i, 0].mean(),
              X_r[r == i, 1].mean(),
              X_r[r == i, 2].mean(), target_name2,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    ax.scatter(X_r[r == i, 0], X_r[r == i, 1], X_r[r == i, 2], color=color, alpha=.8, lw=0.5,
                label=target_name1, cmap=plt.cm.spectral, edgecolor='k')


X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / len(X)
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)


print('3 components', pca.components_)

print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')


plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.tight_layout()
plt.savefig('simple_all_pca_high_medium_low.pdf', transparent=True)
ax.view_init(elev=50., azim=10)
plt.savefig('simple_all_pca_high_medium_low_changed_angle.pdf', transparent=True)

plt.show(block=True)



"""
def compute_scores(X):
    pca = decomposition.PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


for X, title in [(X, 'Heteroscedastic Noise')]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = decomposition.PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()
"""