import csv
from datetime import datetime,timedelta
import logging
import sys
import os

import pandas as pd
from pandas.plotting import andrews_curves

import numpy as np

import scipy
from scipy import interpolate
from scipy.stats import kendalltau

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern, Exponentiation
from sklearn.datasets import fetch_mldata

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
    #"patch.linewidth": 1,
    #"patch.edgecolor": 'k',
    #"patch.force_edgecolor": True,
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
        

# http://statisticsbyjim.com/regression/heteroscedasticity-regression/
# https://cs.stanford.edu/~quocle/LeSmoCan05.pdf
# https://github.com/josephmisiti/awesome-machine-learning
# http://etheses.whiterose.ac.uk/9968/1/Damianou_Thesis.pdf
# ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPythonDraft.pdf
# Resid multi-jointplot all scenarios
key1 = 'RSRP'
key2 = 'Intermediate KPI'
hue = 'Type'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')

rolling_window = 10
datasets1_combined = (pd.concat(datasets1)[['RSRP', 'RSRQ', 'RSSI','Intermediate KPI', 'SINR Rx[0]']].sort_values(by=['RSRP']).rolling(rolling_window).sum() / rolling_window).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
datasets1_combined['Type'] = 'Highway'
datasets2_combined = (pd.concat(datasets2)[['RSRP', 'RSRQ', 'RSSI','Intermediate KPI', 'SINR Rx[0]']].sort_values(by=['RSRP']).rolling(rolling_window).sum() / rolling_window).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
datasets2_combined['Type'] = 'City'
datasets3_combined = (pd.concat(datasets3)[['RSRP', 'RSRQ', 'RSSI','Intermediate KPI', 'SINR Rx[0]']].sort_values(by=['RSRP']).rolling(rolling_window).sum() / rolling_window).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
datasets3_combined['Type'] = 'Country'

datasets123 = [datasets1_combined, datasets2_combined, datasets3_combined]
datasets123_combined = pd.concat(datasets123)


datasets123_combined = datasets123_combined.reset_index(drop=True).sort_values(by=['RSRP'])

#datasets123_combined = datasets123_combined[['RSRP', 'RSRQ', 'RSSI','Intermediate KPI']].rolling(rolling_window).sum() / rolling_window
datasets123_combined = datasets123_combined.dropna(subset=['RSRP', 'RSRQ', 'RSSI', 'SINR Rx[0]','Intermediate KPI'])

datasets123_combined = datasets123_combined[::6]


#datasets123_combined = datasets123_combined[['RSRP', 'RSRQ', 'RSSI','Intermediate KPI']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

X = datasets123_combined[['RSRP']].values
y = datasets123_combined[['Intermediate KPI']].values.ravel()

# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=False)

# Instanciate a Gaussian Process model
#kernel_gpml = Exponentiation(length_scale=1.3, periodicity=1.0)#ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#kernel_gpml =  ExpSineSquared(length_scale=0.215, periodicity=100)
#gp = GaussianProcessRegressor(kernel=kernel_gpml, n_restarts_optimizer=9, alpha=1.0)

gp.fit(X, y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))


# Kernel with optimized parameters
k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))


X_ = np.linspace(X.min(), X.max() * 1.10, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

# Illustration
plt.scatter(X, y, c='#55A868')
plt.plot(datasets1_combined['RSRP'], datasets1_combined['Intermediate KPI'], c="red")
plt.plot(datasets2_combined['RSRP'], datasets2_combined['Intermediate KPI'], c="green")
plt.plot(datasets3_combined['RSRP'], datasets3_combined['Intermediate KPI'], c="blue")
plt.plot(X_, y_pred, c='#4C72B0')
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.2, color='#8172B2')#color='#C44E52')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Measurement")
plt.ylabel("KPI")
plt.title("Gaussian Process (GPML)")
plt.tight_layout()
plt.show()