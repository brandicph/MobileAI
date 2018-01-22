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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
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


def throughput(bandwidth, modulation, mimo):
    # Bandwidth (MHz) to Ressource Blocks
    bandwidth = str(bandwidth)

    # 3GPP TS 36.213 version 10.1.0 Release 10, ETSI TS 136 213 V10.1.0 (2011-04)
    # Table 7.1.7.2.1-1: Transport block size table (dimension 27×110)
    # http://www.etsi.org/deliver/etsi_ts/136200_136299/136213/10.01.00_60/ts_136213v100100p.pdf
    N_PRB_I_TBS = {
        '6': [152,208,256,328,408,504,600,712,808,936,1032,1192,1352,1544,1736,1800,1928,2152,2344,2600,2792,2984,3240,3496,3624,3752,4392],
        '15': [392,520,648,872,1064,1320,1544,1800,2088,2344,2664,2984,3368,3880,4264,4584,4968,5352,5992,6456,6968,7480,7992,8504,9144,9528,11064],
        '25': [680,904,1096,1416,1800,2216,2600,3112,3496,4008,4392,4968,5736,6456,7224,7736,7992,9144,9912,10680,11448,12576,13536,14112,15264,15840,18336],
        '50': [1384,1800,2216,2856,3624,4392,5160,6200,6968,7992,8760,9912,11448,12960,14112,15264,16416,18336,19848,21384,22920,25456,27376,28336,30576,31704,36696],
        '75': [2088,2728,3368,4392,5352,6712,7736,9144,10680,11832,12960,15264,16992,19080,21384,22920,24496,27376,29296,32856,35160,37888,40576,43816,45352,46888,55056],
        '100': [2792,3624,4584,5736,7224,8760,10296,12216,14112,15840,17568,19848,22920,25456,28336,30576,32856,36696,39232,43816,46888,51024,55056,57336,61664,63776,75376]
    }

    # ftp://www.3gpp.org/workshop/2009-12-17_ITU-R_IMT-Adv_eval/docs/pdf/REV-090003-r1.pdf
    # http://www.etsi.org/deliver/etsi_ts/136100_136199/136116/12.04.00_60/ts_136116v120400p.pdf
    # Table 6.5.3.1-3: Relay operating band unwanted emission limits for 5, 10, 15 and 20 MHz channel bandwidth 
    BW_RB = {
        '1.4': N_PRB_I_TBS['6'],
        '3': N_PRB_I_TBS['15'],
        '5': N_PRB_I_TBS['25'],
        '10': N_PRB_I_TBS['50'],
        '15': N_PRB_I_TBS['75'],
        '20': N_PRB_I_TBS['100']
    }

    # Table 7.1.7.1-1: Modulation and TBS index table for PDSCH 
    # QPSK: 2
    # 16QAM: 4
    # 64QAM: 6
    # 256QAM: 8
    Modulation_TBS = {
        'QPSK': 9, # Best: I_MCS=9, I_TBS=9
        '16QAM': 15, # Best: I_MCS=16, I_TBS=15
        '64QAM': 26, # Best: I_MCS=28, I_TBS=26
        '256QAM': 33 # Best: I_MCS=?, I_TBS=33
    }

    MIMO = {
        'SISO 1x1': 1,
        'MIMO 2x2': 2,
        'MIMO 4x4': 4
    }

    #BW_MCS_Code_Rate = {}

    return BW_RB[bandwidth][Modulation_TBS[modulation]] * MIMO[mimo] / 1000

bitrate = throughput(20, '64QAM', 'MIMO 2x2')

print(bitrate)

plt.figure()

plt.plot([2088,2728,3368,4392,5352,6712,7736,9144,10680,11832,12960,15264,16992,19080,21384,22920,24496,27376,29296,32856,35160,37888,40576,43816,45352,46888,55056], label='15 MHz')
plt.plot([2792,3624,4584,5736,7224,8760,10296,12216,14112,15840,17568,19848,22920,25456,28336,30576,32856,36696,39232,43816,46888,51024,55056,57336,61664,63776,75376], label='20 MHz')

plt.legend()
plt.show()

DATASET_TYPE = 'highway'
datasets = load_dataset(type=DATASET_TYPE)
#print(datasets)

# Sort values
"""
df12_combined = df12_combined.sort_values(by=['Intermediate KPI_x'])
df22_combined = df22_combined.sort_values(by=['Intermediate KPI_x'])
df32_combined = df32_combined.sort_values(by=['Intermediate KPI_x'])
df42_combined = df42_combined.sort_values(by=['Intermediate KPI_x'])
df52_combined = df52_combined.sort_values(by=['Intermediate KPI_x'])
"""
# Perform moving average
"""
window = 5
df12_combined = df12_combined.rolling(window).sum()/window
df22_combined = df22_combined.rolling(window).sum()/window
df32_combined = df32_combined.rolling(window).sum()/window
df42_combined = df42_combined.rolling(window).sum()/window
df52_combined = df52_combined.rolling(window).sum()/window
"""

"""
# Combined PairGrid
datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')
datasets123 = datasets1 + datasets2 + datasets3
datasets123_combined = pd.concat(datasets123)

datasets_combined = datasets123_combined#pd.concat(datasets)
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(datasets_combined[['RSRP', 'RSSI', 'SINR Rx[0]', 'Intermediate KPI', 'Average MCS Index', 'Num QPSK (LTE DL)', 'Num 16QAM (LTE DL)', 'Num 64QAM (LTE DL)']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")

plt.tight_layout()
plt.savefig('modulation_{}_pairplot.pdf'.format(DATASET_TYPE))

plt.show(block=False)
"""

# Lmplot
key1 = 'Average MCS Index'
key2 = 'Intermediate KPI'
hue = 'Type'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')

datasets1_combined = pd.concat(datasets1)
datasets1_combined['Type'] = 'Highway'
datasets2_combined = pd.concat(datasets2)
datasets2_combined['Type'] = 'City'
datasets3_combined = pd.concat(datasets3)
datasets3_combined['Type'] = 'Country'

datasets123 = [datasets1_combined, datasets2_combined, datasets3_combined]
datasets123_combined = pd.concat(datasets123)

sns.lmplot(x=key1, y=key2, data=datasets123_combined, hue=hue, legend=False, scatter_kws=dict(edgecolor='k', linewidth=1));

#plt.ylim(ylim_rsrp_kpi)
#plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('modulation_{}_lmplot_regression.pdf'.format('all'))

plt.show(block=True)