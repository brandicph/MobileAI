import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression

#https://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python

import seaborn as sns

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
datasets123_combined.dropna(subset=['RSRP', 'RSRP Rx[0]', 'RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRQ', 'Intermediate KPI'],inplace=True)



df = datasets123_combined
X = df[['RSRP', 'RSRP Rx[0]', 'RSRP Rx[1]', 'SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRQ']]
#X = df[['RSRP', 'SINR Rx[0]', 'RSSI', 'RSRQ']]
y = df['Intermediate KPI']

pca = PCA()

X_reduced = pca.fit_transform(scale(X))

variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(variance)



n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

regr = LinearRegression()
mse = []

score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score) 


for i in np.arange(1,6):
    score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], y.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(mse, '-v')
ax2.plot([1,2,3,4,5], mse[1:6], '-v')
ax2.set_title('Intercept excluded from plot')

for ax in fig.axes:
    ax.set_xlabel('Number of principal components in regression')
    ax.set_ylabel('MSE')
    ax.set_xlim((-0.2,5.2))


plt.show()