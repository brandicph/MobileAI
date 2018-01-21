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
        

DATASET_TYPE = 'highway'
datasets = load_dataset(type=DATASET_TYPE)
print(datasets)

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

# One column
fig, ax1 = plt.subplots()
key = 'RSRP'

for index, dataset in enumerate(datasets):
    column = dataset[key]
    ax1.plot(np.arange(len(column)), column, label='Test {}'.format(index+1))

ax1.set_xlabel('Measurement')
ax1.set_ylabel('RSRP (dBm)')
ax1.legend(loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
fig.tight_layout()
plt.savefig('simple_{}_line_rsrp.pdf'.format(DATASET_TYPE))
plt.show(block=False)
#fig.savefig("lineplot_sinr_raw.pdf")


# Another column
fig, ax1 = plt.subplots()
key = 'Intermediate KPI'

for index, dataset in enumerate(datasets):
    column = dataset[key]
    ax1.plot(np.arange(len(column)), column, label='Test {}'.format(index+1))

ax1.set_xlabel('Measurement')
ax1.set_ylabel('Intermediate KPI (kbps)')
ax1.legend(loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
#autolabel(rects,ax1)
#ax1.set_ylim((0, 200))
fig.tight_layout()
plt.savefig('simple_{}_line_kpi.pdf'.format(DATASET_TYPE))
plt.show(block=False)


# Line combined merged
fig, ax1 = plt.subplots()
key = 'Intermediate KPI'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')
datasets1_combined = pd.concat(datasets1)
datasets2_combined = pd.concat(datasets2)
datasets3_combined = pd.concat(datasets3)
s1 = datasets1_combined[key]
s2 = datasets2_combined[key]
s3 = datasets3_combined[key]
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('Intermediate KPI (kbps)')
ax1.legend(loc='upper right')
ax1.tick_params('y')
ax1.legend(['Highway', 'City', 'Country'], loc='upper right')
fig.tight_layout()
plt.savefig('simple_{}_line_kpi_combined.pdf'.format(DATASET_TYPE))
plt.show(block=False)


# Line combined NOT merged
fig, ax1 = plt.subplots()
key = 'RSRP'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')
datasets1_combined = pd.concat(datasets1)
datasets2_combined = pd.concat(datasets2)
datasets3_combined = pd.concat(datasets3)
s1 = datasets1_combined[key]
s2 = datasets2_combined[key]
s3 = datasets3_combined[key]
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('RSRP (dBm)')
ax1.legend(loc='upper right')
ax1.tick_params('y')
ax1.legend(['Highway', 'City', 'Country'], loc='upper right')
fig.tight_layout()
plt.savefig('simple_{}_line_rsrp_combined.pdf'.format(DATASET_TYPE))
plt.show(block=False)

"""
# Single PairGrid
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(datasets[1][['RSRP', 'SINR Rx[0]', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.tight_layout()
plt.savefig('simple_{}_pairgrid_reference.pdf'.format(DATASET_TYPE))
plt.show(block=False)


# Single PairGrid
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(datasets[1][['RSRP', 'SINR Rx[0]', 'Bytes Transferred', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.tight_layout()
plt.savefig('simple_{}_pairgrid_mixed.pdf'.format(DATASET_TYPE))
plt.show(block=False)


# Single PairGrid Extended
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(datasets[1][['RSRP', 'RSRQ', 'RSSI', 'SINR Rx[0]', 'SINR Rx[1]', 'BLER', 'Bytes Transferred', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.tight_layout()
plt.savefig('simple_{}_pairgrid_reference_extended.pdf'.format(DATASET_TYPE))
plt.show(block=False)
"""

"""
# Combined PairGrid
datasets_combined = pd.concat(datasets)
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(datasets_combined[['RSRP', 'RSSI', 'SINR Rx[0]', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.show(block=False)
"""

ylim_rsrp_kpi = (-10000, 150000)
xlim_rsrp_kpi = (-120, -50)
# Joint plot two values
g = sns.jointplot('RSRP', 'Intermediate KPI', data=datasets[0], kind='reg', color="#990000", scatter_kws=dict(s=50, edgecolor='k', linewidth=1), marginal_kws=dict(color='grey'), line_kws=dict(lw=2, color="black"))
plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')


xdata = datasets[0]['RSRP']
ydata = datasets[0]['Intermediate KPI']

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=xdata,y=ydata)
print('Single Test', "slope: {:20} intercept: {:20} r_value: {:20} p_value: {:20} std_err: {:20}".format(slope, intercept, r_value, p_value, std_err))

plt.ylim(ylim_rsrp_kpi)
plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_jointplot_rsrp_kpi.pdf'.format(DATASET_TYPE))
plt.show(block=False)


# Combined multi-jointplot
key1 = 'RSRP'
key2 = 'Intermediate KPI'

datasets_combined = pd.concat(datasets)

graph = sns.jointplot(key1, key2, data=datasets_combined, kind='reg', marker='.', color="grey", scatter_kws=dict(s=1), line_kws=dict(lw=2, color="black"))

xdata = datasets_combined[key1]
ydata = datasets_combined[key2]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=xdata,y=ydata)
print('All Tests', "slope: {:20} intercept: {:20} r_value: {:20} p_value: {:20} std_err: {:20}".format(slope, intercept, r_value, p_value, std_err))

graph.x = datasets[0][key1]
graph.y = datasets[0][key2]
graph.plot_joint(plt.scatter, c='#c0392b', s=50, label='Test 1', edgecolor='k', linewidth=1)

graph.x = datasets[1][key1]
graph.y = datasets[1][key2]
graph.plot_joint(plt.scatter, c='#7f8c8d', s=50, label='Test 2', edgecolor='k', linewidth=1)

graph.x = datasets[2][key1]
graph.y = datasets[2][key2]
graph.plot_joint(plt.scatter, c='#2c3e50', s=50, label='Test 3', edgecolor='k', linewidth=1)

graph.x = datasets[3][key1]
graph.y = datasets[3][key2]
graph.plot_joint(plt.scatter, c='#8e44ad', s=50, label='Test 4', edgecolor='k', linewidth=1)

graph.x = datasets[4][key1]
graph.y = datasets[4][key2]
graph.plot_joint(plt.scatter, c='#16a085', s=50, label='Test 5', edgecolor='k', linewidth=1)

plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')

plt.ylim(ylim_rsrp_kpi)
plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_jointplot_rsrp_kpi_combined.pdf'.format(DATASET_TYPE))

plt.show(block=False)


# Combined multi-jointplot (City)
key1 = 'RSRP'
key2 = 'Intermediate KPI'

data_custom = 'city'
datasets_custom = load_dataset(type=data_custom)
datasets_custom_combined = pd.concat(datasets_custom)

graph = sns.jointplot(key1, key2, data=datasets_custom_combined, kind='reg', marker='.', color="grey", scatter_kws=dict(s=1), line_kws=dict(lw=2, color="black"))

graph.x = datasets_custom[0][key1]
graph.y = datasets_custom[0][key2]
graph.plot_joint(plt.scatter, c='#c0392b', s=50, label='Test 1', edgecolor='k', linewidth=1)

graph.x = datasets_custom[1][key1]
graph.y = datasets_custom[1][key2]
graph.plot_joint(plt.scatter, c='#7f8c8d', s=50, label='Test 2', edgecolor='k', linewidth=1)

graph.x = datasets_custom[2][key1]
graph.y = datasets_custom[2][key2]
graph.plot_joint(plt.scatter, c='#2c3e50', s=50, label='Test 3', edgecolor='k', linewidth=1)

graph.x = datasets_custom[3][key1]
graph.y = datasets_custom[3][key2]
graph.plot_joint(plt.scatter, c='#8e44ad', s=50, label='Test 4', edgecolor='k', linewidth=1)

graph.x = datasets_custom[4][key1]
graph.y = datasets_custom[4][key2]
graph.plot_joint(plt.scatter, c='#16a085', s=50, label='Test 5', edgecolor='k', linewidth=1)

plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')

plt.ylim(ylim_rsrp_kpi)
plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_jointplot_rsrp_kpi_combined.pdf'.format(data_custom))

plt.show(block=False)



# Combined multi-jointplot all scenarios
key1 = 'RSRP'
key2 = 'Intermediate KPI'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')
datasets123 = datasets1 + datasets2 + datasets3
datasets123_combined = pd.concat(datasets123)

datasets1_combined = pd.concat(datasets1)
datasets2_combined = pd.concat(datasets2)
datasets3_combined = pd.concat(datasets3)

graph = sns.jointplot(key1, key2, data=datasets123_combined, kind='reg', marker='.', color="grey", scatter_kws=dict(s=1), line_kws=dict(lw=2, color="black"))

xdata = datasets123_combined[key1]
ydata = datasets123_combined[key2]

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=xdata,y=ydata)
print('All Scenarios', "slope: {:20} intercept: {:20} r_value: {:20} p_value: {:20} std_err: {:20}".format(slope, intercept, r_value, p_value, std_err))

graph.x = datasets1_combined[key1]
graph.y = datasets1_combined[key2]
graph.plot_joint(plt.scatter, c='#c0392b', s=50, label='Highway', edgecolor='k', linewidth=1)

graph.x = datasets2_combined[key1]
graph.y = datasets2_combined[key2]
graph.plot_joint(plt.scatter, c='#7f8c8d', s=50, label='City', edgecolor='k', linewidth=1)

graph.x = datasets3_combined[key1]
graph.y = datasets3_combined[key2]
graph.plot_joint(plt.scatter, c='#2c3e50', s=50, label='Country', edgecolor='k', linewidth=1)

plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')

plt.ylim(ylim_rsrp_kpi)
plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_jointplot_rsrp_kpi_combined_scenarios.pdf'.format(DATASET_TYPE))

plt.show(block=False)


#f, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
f, (ax1, ax2, ax3) = plt.subplots(3)
# http://statisticsbyjim.com/regression/heteroscedasticity-regression/
# https://cs.stanford.edu/~quocle/LeSmoCan05.pdf
# https://github.com/josephmisiti/awesome-machine-learning
# http://etheses.whiterose.ac.uk/9968/1/Damianou_Thesis.pdf
# ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPythonDraft.pdf
# Resid multi-jointplot all scenarios
key1 = 'RSRP'
key2 = 'Intermediate KPI'

datasets1 = load_dataset(type='highway')
datasets2 = load_dataset(type='city')
datasets3 = load_dataset(type='308')
datasets123 = datasets1 + datasets2 + datasets3
datasets123_combined = pd.concat(datasets123)

datasets1_combined = pd.concat(datasets1)
datasets2_combined = pd.concat(datasets2)
datasets3_combined = pd.concat(datasets3)

sns.residplot(x=key1, y=key2, data=datasets1_combined, ax=ax1, label='Highway', scatter_kws=dict(edgecolor='k', linewidth=1));
ax1.legend(['y = 0', 'Highway'])
sns.residplot(x=key1, y=key2, data=datasets2_combined, ax=ax2, label='City', scatter_kws=dict(edgecolor='k', linewidth=1));
ax2.legend(['y = 0', 'City'])
sns.residplot(x=key1, y=key2, data=datasets3_combined, ax=ax3, label='Country', scatter_kws=dict(edgecolor='k', linewidth=1));
ax3.legend(['y = 0', 'Country'])

plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')
plt.tight_layout()
plt.savefig('simple_{}_residual_rsrp_kpi.pdf'.format('all'))

plt.show(block=False)



# Combined multi-jointplot (Highway)
key1 = 'RSRP'
key2 = 'Intermediate KPI'

data_custom = 'highway'
hue = 'Type'
datasets_custom = load_dataset(type=data_custom)
datasets_custom[0]['Type'] = 'Test 1'
datasets_custom[1]['Type'] = 'Test 2'
datasets_custom[2]['Type'] = 'Test 3'
datasets_custom[3]['Type'] = 'Test 4'
datasets_custom[4]['Type'] = 'Test 5'

datasets_custom_combined = pd.concat(datasets_custom)

sns.lmplot(x=key1, y=key2, data=datasets_custom_combined, hue=hue, legend=False, scatter_kws=dict(edgecolor='k', linewidth=1));


plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')

plt.ylim(ylim_rsrp_kpi)
plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_lmplot_rsrp_kpi_tests_seperate.pdf'.format(data_custom))

plt.show(block=False)


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

datasets1_combined = pd.concat(datasets1)
datasets1_combined['Type'] = 'Highway'
datasets2_combined = pd.concat(datasets2)
datasets2_combined['Type'] = 'City'
datasets3_combined = pd.concat(datasets3)
datasets3_combined['Type'] = 'Country'

datasets123 = [datasets1_combined, datasets2_combined, datasets3_combined]
datasets123_combined = pd.concat(datasets123)

sns.lmplot(x=key1, y=key2, data=datasets123_combined, hue=hue, legend=False, scatter_kws=dict(edgecolor='k', linewidth=1));

plt.xlabel('RSRP (dBm)')
plt.ylabel('Intermediate KPI (kbps)')

#plt.ylim(ylim_rsrp_kpi)
#plt.xlim(xlim_rsrp_kpi)

plt.legend()
plt.tight_layout()
plt.savefig('simple_{}_lmplot_rsrp_kpi_regression.pdf'.format('all'))

plt.show(block=True)