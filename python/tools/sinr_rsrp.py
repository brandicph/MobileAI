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


# CSV file path
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Airport Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_airport_ping2.csv')
# DTU 1GB download
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# Instatiate data reader
qp = QualiPoc(CSV_IN_FILEPATH)

# Perform filtering
n_samples_before_filtering = len(qp.df)
qp.logger.info("Before filtering: {} samples".format(n_samples_before_filtering))
df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

df = df#[df['Intermediate KPI'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))

# CSV file path
fig, ax1 = plt.subplots()
s10 = df['SINR Rx[0]']
s11 = df['SINR Rx[1]']
ax1.plot(s10, color='#4C72B0')
ax1.plot(s11, color='#55A868')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[0] / Rx[1] (dB)')
ax1.legend(['SINR Rx[0]', 'SINR Rx[1]'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)

ax2 = ax1.twinx()
s20 = df['RSRP Rx[0]']
s21 = df['RSRP Rx[1]']
ax2.plot(s20, color='#C44E52')
ax2.plot(s21, color='#8172B2')
ax2.set_ylabel('RSRP Rx[0] / Rx[1] (dBm)')
ax2.legend(['RSRP Rx[0]', 'RSRP Rx[1]'], loc='upper right', bbox_to_anchor=(1.0, 0.95), ncol=2)
ax2.tick_params('y')
#ax2.set_yscale('log', basey=2)
ax2.grid(False)
fig.tight_layout()

plt.show(block=False)

# Perform rolling mean
df_sinr_rsrp_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]']].rolling(200, win_type='triang').sum()/200

# SINR0 / RSRP0
fig, ax1 = plt.subplots()
s10 = df_sinr_rsrp_rolling['SINR Rx[0]']
ax1.plot(s10, color='#4C72B0')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[0] (dB)')
ax1.legend(['SINR Rx[0]'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)

ax2 = ax1.twinx()
s20 = df_sinr_rsrp_rolling['RSRP Rx[0]']
ax2.plot(s20, color='#55A868')
ax2.set_ylabel('RSRP Rx[0] (dBm)')
ax2.legend(['RSRP Rx[0]'], loc='upper right', bbox_to_anchor=(1.0, 0.95), ncol=2)
ax2.tick_params('y')
#ax2.set_yscale('log', basey=2)
ax2.grid(False)
fig.tight_layout()

plt.show(block=False)

# SINR1 / RSRP1
fig, ax1 = plt.subplots()
s10 = df_sinr_rsrp_rolling['SINR Rx[1]']
ax1.plot(s10, color='#C44E52')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[1] (dB)')
ax1.legend(['SINR Rx[1]'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)

ax2 = ax1.twinx()
s20 = df_sinr_rsrp_rolling['RSRP Rx[1]']
ax2.plot(s20, color='#8172B2')
ax2.set_ylabel('RSRP Rx[1] (dBm)')
ax2.legend(['RSRP Rx[1]'], loc='upper right', bbox_to_anchor=(1.0, 0.95), ncol=2)
ax2.tick_params('y')
#ax2.set_yscale('log', basey=2)
ax2.grid(False)
fig.tight_layout()


# CSV file path
fig, ax1 = plt.subplots()
s10 = df['SINR Rx[0]']
s11 = df['SINR Rx[1]']
ax1.plot(s10, color='#333333')
ax1.plot(s11, color='#C44E52')
ax1.set_xlabel('Measurement')
ax1.set_ylabel('SINR (dB)')
ax1.legend(['SINR $Rx_0$', 'SINR $Rx_1$'], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
fig.tight_layout()
plt.show(block=False)
fig.savefig("lineplot_sinr_raw.pdf")

fig, ax1 = plt.subplots()
s20 = df['RSRP Rx[0]']
s21 = df['RSRP Rx[1]']
ax1.plot(s20, color='#333333')
ax1.plot(s21, color='#C44E52')
ax1.set_ylabel('RSRP (dBm)')
ax1.set_xlabel('Measurement')
ax1.legend(['RSRP $Rx_0$', 'RSRP $Rx_1$'], loc='upper right')
ax1.tick_params('y')
#ax2.set_yscale('log', basey=2)
fig.tight_layout()
plt.show(block=False)
fig.savefig("lineplot_rsrp_raw.pdf")



def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


fig, ax1 = plt.subplots()
s10 = df#[df['Intermediate KPI'] > 0]
s10 = s10['Intermediate KPI']
ind = np.arange(len(s10))
rects = ax1.plot(ind, s10, color='#C44E52')
ax1.set_xlabel('Measurement')
ax1.set_ylabel('Intermediate KPI (kbps)')
ax1.legend(['Intermediate KPI (RTT)'], loc='upper right')
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)
#autolabel(rects,ax1)
#ax1.set_ylim((0, 200))
fig.tight_layout()
plt.show(block=False)
fig.savefig("lineplot_intermediate_kpi_raw.pdf")



#plt.title('SINR vs. RSRP\nwithout preprocessing')
#['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
#print(mpl.rcParams['axes.prop_cycle'])

qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')