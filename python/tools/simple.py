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


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
CSV001 = os.path.join(SCRIPT_PATH, '../../data/308/1.csv')
CSV002 = os.path.join(SCRIPT_PATH, '../../data/308/2.csv')
CSV003 = os.path.join(SCRIPT_PATH, '../../data/308/3.csv')
CSV004 = os.path.join(SCRIPT_PATH, '../../data/308/4.csv')
CSV005 = os.path.join(SCRIPT_PATH, '../../data/308/5.csv')

df1 = pd.read_csv(CSV001)
df2 = pd.read_csv(CSV002)
df3 = pd.read_csv(CSV003)
df4 = pd.read_csv(CSV004)
df5 = pd.read_csv(CSV005)

# CSV file path
fig, ax1 = plt.subplots()
s1 = df1['RSRP']
s2 = df2['RSRP']
s3 = df3['RSRP']
s4 = df4['RSRP']
s5 = df5['RSRP']
ax1.plot(s1)
ax1.plot(s2)
ax1.plot(s3)
ax1.plot(s4)
ax1.plot(s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('RSRP (dB)')
ax1.legend(['RSRP 1', 'RSRP 2', 'RSRP 3', 'RSRP 4', 'RSRP 5'], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
fig.tight_layout()
plt.show(block=False)
#fig.savefig("lineplot_sinr_raw.pdf")


fig, ax1 = plt.subplots()
s1 = df1['Intermediate KPI']
s2 = df2['Intermediate KPI']
s3 = df3['Intermediate KPI']
s4 = df4['Intermediate KPI']
s5 = df5['Intermediate KPI']
ax1.plot(s1)
ax1.plot(s2)
ax1.plot(s3)
ax1.plot(s4)
ax1.plot(s5)
ax1.set_xlabel('Measurement')
ax1.set_ylabel('kbps')
ax1.legend(['Intermediate KPI 1', 'Intermediate KPI 2', 'Intermediate KPI 3', 'Intermediate KPI 4', 'Intermediate KPI 5'], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
#autolabel(rects,ax1)
#ax1.set_ylim((0, 200))
fig.tight_layout()
plt.show(block=True)
