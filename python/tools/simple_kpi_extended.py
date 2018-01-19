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
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn.datasets import fetch_mldata

mpl.style.use('seaborn')

pgf_with_custom_preamble = {
    "font.family": 'serif',
    "font.serif": 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
    "font.sans-serif": 'Helvetica, Avant Garde, Computer Modern Sans serif',
    "font.cursive": 'Zapf Chancery',
    "font.monospace": 'Courier, Computer Modern Typewriter',
    "text.usetex": True,
    "text.dvipnghack": True,
    "axes.color_cycle": ['#c0392b', '#7f8c8d', '#2c3e50', '#8e44ad', '#16a085']
}
mpl.rcParams.update(pgf_with_custom_preamble)


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
DATA_PATH = os.path.join(SCRIPT_PATH, '../../data/kpi/highway/')
DATA_PATH2 = os.path.join(SCRIPT_PATH, '../../data/signal_quality/highway/')

# Path for first dataset
CSV001 = os.path.join(DATA_PATH, '1.csv')
CSV002 = os.path.join(DATA_PATH, '2.csv')
CSV003 = os.path.join(DATA_PATH, '3.csv')
CSV004 = os.path.join(DATA_PATH, '4.csv')
CSV005 = os.path.join(DATA_PATH, '5.csv')

# Path for second dataset
CSV0012 = os.path.join(DATA_PATH2, '1.csv')
CSV0022 = os.path.join(DATA_PATH2, '2.csv')
CSV0032 = os.path.join(DATA_PATH2, '3.csv')
CSV0042 = os.path.join(DATA_PATH2, '4.csv')
CSV0052 = os.path.join(DATA_PATH2, '5.csv')

# Read first dataset
df1 = pd.read_csv(CSV001)
df2 = pd.read_csv(CSV002)
df3 = pd.read_csv(CSV003)
df4 = pd.read_csv(CSV004)
df5 = pd.read_csv(CSV005)

# Set index for first dataset
df1.set_index('Time', inplace=True)
df2.set_index('Time', inplace=True)
df3.set_index('Time', inplace=True)
df4.set_index('Time', inplace=True)
df5.set_index('Time', inplace=True)

# Read second dataset
df12 = pd.read_csv(CSV0012)
df22 = pd.read_csv(CSV0022)
df32 = pd.read_csv(CSV0032)
df42 = pd.read_csv(CSV0042)
df52 = pd.read_csv(CSV0052)

# Set index for second dataset
df12.set_index('Time', inplace=True)
df22.set_index('Time', inplace=True)
df32.set_index('Time', inplace=True)
df42.set_index('Time', inplace=True)
df52.set_index('Time', inplace=True)

# Merge by first dataset index
df12_combined = pd.merge(df1, df12, left_index=True, right_index=True, how='inner')
df22_combined = pd.merge(df2, df22, left_index=True, right_index=True, how='inner')
df32_combined = pd.merge(df3, df32, left_index=True, right_index=True, how='inner')
df42_combined = pd.merge(df4, df42, left_index=True, right_index=True, how='inner')
df52_combined = pd.merge(df5, df52, left_index=True, right_index=True, how='inner')

# Rename columns after merge
df12_combined.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)
df22_combined.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)
df32_combined.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)
df42_combined.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)
df52_combined.rename(columns={'Intermediate KPI_x': 'Intermediate KPI'}, inplace=True)

# Drop NaN
df12_combined.dropna(subset=['Intermediate KPI'], inplace=True)
df22_combined.dropna(subset=['Intermediate KPI'], inplace=True)
df32_combined.dropna(subset=['Intermediate KPI'], inplace=True)
df42_combined.dropna(subset=['Intermediate KPI'], inplace=True)
df52_combined.dropna(subset=['Intermediate KPI'], inplace=True)

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

# CSV file path
fig, ax1 = plt.subplots()
key = 'RSRP'
s1 = df12_combined[key]
s2 = df22_combined[key]
s3 = df32_combined[key]
s4 = df42_combined[key]
s5 = df52_combined[key]
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.plot(np.arange(len(s4)), s4)
ax1.plot(np.arange(len(s5)), s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('dB')
ax1.legend(loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
fig.tight_layout()
plt.show(block=False)
#fig.savefig("lineplot_sinr_raw.pdf")

fig, ax1 = plt.subplots()
key = 'Intermediate KPI'
s1 = df12_combined[key]
s2 = df22_combined[key]
s3 = df32_combined[key]
s4 = df42_combined[key]
s5 = df52_combined[key]
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.plot(np.arange(len(s4)), s4)
ax1.plot(np.arange(len(s5)), s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('kbps')
ax1.legend(loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
#autolabel(rects,ax1)
#ax1.set_ylim((0, 200))
fig.tight_layout()
plt.show(block=False)


# Single PairGrid
cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(df12_combined[['RSRP', 'RSRQ', 'RSSI', 'SINR Rx[0]', 'Bytes Transferred', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.show(block=False)


"""
# Combined PairGrid
frames = [df12_combined, df22_combined, df32_combined, df42_combined, df52_combined]
result = pd.concat(frames)

cmap = mpl.colors.ListedColormap(['#cc5155', '#C44E52', '#873638', '#602527'])
g = sns.PairGrid(result[['RSRP', 'RSRQ', 'RSSI', 'SINR Rx[0]', 'Bytes Transferred', 'Intermediate KPI']])
g = g.map_lower(sns.regplot, color="#C44E52", marker='+', scatter_kws=dict(s=15), line_kws=dict(lw=2, color="black"))
g = g.map_upper(sns.kdeplot, cmap=cmap, shade=False, shade_lowest=False)
g = g.map_diag(sns.kdeplot, legend=False, shade=True, color="black")
plt.show(block=False)
"""

"""
g = sns.jointplot('RSRP', 'Intermediate KPI', data=df12_combined, kind='reg')
plt.show(block=True)
"""