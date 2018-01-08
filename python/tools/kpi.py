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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from pandas.plotting import andrews_curves

from qp import QualiPoc
mpl.style.use('seaborn')


# CSV file path
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
# S-TOG Ping (DTU)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
# S-TOG Ping (Bx)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-04-08-41-51-0000-5310-7746-0004-S_ping2.csv')
# Airport Ping
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_airport_ping2.csv')
# City Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-17-11-58-0000-5310-7746-0004-S_city_ping2.csv')

# Instatiate data reader
qp = QualiPoc(CSV_IN_FILEPATH)

# Perform filtering
n_samples_before_filtering = len(qp.df)
qp.logger.info("Before filtering: {} samples".format(n_samples_before_filtering))
df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

#df = df[df['Intermediate KPI'] > 0]
#df = df[df['SINR Rx[0]'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))

# Perform rolling mean
rolling_window = 100
df_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'Intermediate KPI']].rolling(rolling_window).sum() / rolling_window

df_rolling = df_rolling.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])


# SINR0 / KPI
"""
fig, ax1 = plt.subplots()
s10 = df_rolling['SINR Rx[0]']
s11 = df_rolling['Intermediate KPI']
print(s11)
ax1.plot(s10, color='#4C72B0')
ax1.plot(s11, color='#55A868')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[0] (dB) / Intermediate KPI')
ax1.legend(['SINR Rx[0]', 'Intermediate KPI'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1)
ax1.tick_params('y')

fig.tight_layout()

plt.show(block=False)
"""

deg_trend = 20
y_trend0 = df_rolling['SINR Rx[0]']
x_trend0 = pd.Series(np.arange(len(y_trend0)))
z_trend0 = np.polyfit(x=x_trend0, y=y_trend0, deg=deg_trend)
p_trend0 = np.poly1d(z_trend0)

y_trend1 = df_rolling['Intermediate KPI']
x_trend1 = pd.Series(np.arange(len(y_trend1)))
z_trend1 = np.polyfit(x=x_trend1, y=y_trend1, deg=deg_trend)
p_trend1 = np.poly1d(z_trend1)

fig, ax1 = plt.subplots()
s10 = df_rolling['SINR Rx[0]']
ax1.plot(s10, color='#4C72B0')
ax1.plot(x_trend0, p_trend0(x_trend0), color='black')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[0] (dB)')
ax1.legend(['SINR Rx[0]','trend'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)

ax2 = ax1.twinx()
s20 = df_rolling['Intermediate KPI']
ax2.plot(s20, color='#55A868')
ax2.plot(x_trend1, p_trend1(x_trend1), color='red')
ax2.set_ylabel('RTT (ms)')
ax2.legend(['RTT', 'Trend'], loc='upper right', bbox_to_anchor=(1.0, 0.95), ncol=2)
ax2.tick_params('y')
#ax2.set_yscale('log', basey=2)
ax2.grid(False)
fig.tight_layout()

plt.show(block=False)


deg_trend = 20
y_trend0 = df_rolling['SINR Rx[1]']
x_trend0 = pd.Series(np.arange(len(y_trend0)))
z_trend0 = np.polyfit(x=x_trend0, y=y_trend0, deg=deg_trend)
p_trend0 = np.poly1d(z_trend0)

y_trend1 = df_rolling['Intermediate KPI']
x_trend1 = pd.Series(np.arange(len(y_trend1)))
z_trend1 = np.polyfit(x=x_trend1, y=y_trend1, deg=deg_trend)
p_trend1 = np.poly1d(z_trend1)


fig, ax1 = plt.subplots()
s10 = df_rolling['SINR Rx[1]']
ax1.plot(s10, color='#C44E52')
ax1.plot(x_trend0, p_trend0(x_trend0), color='black')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[1] (dB)')
ax1.legend(['SINR Rx[1]', 'Trend'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)

ax2 = ax1.twinx()
s20 = df_rolling['Intermediate KPI']
ax2.plot(s20, color='#8172B2')
ax2.plot(x_trend1, p_trend1(x_trend1), color='red')
ax2.set_ylabel('RTT (ms)')
ax2.legend(['RTT', 'Trend'], loc='upper right', bbox_to_anchor=(1.0, 0.95), ncol=2)
ax2.tick_params('y')
#ax2.set_yscale('log', basey=2)
ax2.grid(False)
fig.tight_layout()


qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')