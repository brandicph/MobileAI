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
DATA_PATH2 = os.path.join(SCRIPT_PATH, '../../data/rsrp_kpi/highway/')

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

# Sort values
df12_combined = df12_combined.sort_values(by=['Intermediate KPI_x'])
df22_combined = df22_combined.sort_values(by=['Intermediate KPI_x'])
df32_combined = df32_combined.sort_values(by=['Intermediate KPI_x'])
df42_combined = df42_combined.sort_values(by=['Intermediate KPI_x'])
df52_combined = df52_combined.sort_values(by=['Intermediate KPI_x'])

# Perform moving average
window = 5
df12_combined = df12_combined.rolling(window).sum()/window
df22_combined = df22_combined.rolling(window).sum()/window
df32_combined = df32_combined.rolling(window).sum()/window
df42_combined = df42_combined.rolling(window).sum()/window
df52_combined = df52_combined.rolling(window).sum()/window

# Print length to ensure same length
print(len(df12_combined), len(df1))

# CSV file path
fig, ax1 = plt.subplots()
s1 = df12_combined['RSRP']
s2 = df22_combined['RSRP']
s3 = df32_combined['RSRP']
s4 = df42_combined['RSRP']
s5 = df52_combined['RSRP']
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.plot(np.arange(len(s4)), s4)
ax1.plot(np.arange(len(s5)), s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('RSRP (dB)')
ax1.legend(['RSRP 1', 'RSRP 2', 'RSRP 3', 'RSRP 4', 'RSRP 5'], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
fig.tight_layout()
plt.show(block=False)
#fig.savefig("lineplot_sinr_raw.pdf")

fig, ax1 = plt.subplots()
print(df12_combined)
s1 = df12_combined['Intermediate KPI_x']
s2 = df22_combined['Intermediate KPI_x']
s3 = df32_combined['Intermediate KPI_x']
s4 = df42_combined['Intermediate KPI_x']
s5 = df52_combined['Intermediate KPI_x']
ax1.plot(np.arange(len(s1)), s1)
ax1.plot(np.arange(len(s2)), s2)
ax1.plot(np.arange(len(s3)), s3)
ax1.plot(np.arange(len(s4)), s4)
ax1.plot(np.arange(len(s5)), s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('kbps')
ax1.legend(['Intermediate KPI 1', 'Intermediate KPI 2', 'Intermediate KPI 3', 'Intermediate KPI 4', 'Intermediate KPI 5', ], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
#autolabel(rects,ax1)
#ax1.set_ylim((0, 200))
fig.tight_layout()
plt.show(block=True)
