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
# Airport Signal Quality
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_signal_quality.csv')
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
df_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'RSRQ Rx[0]', 'RSSI Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'RSRQ Rx[1]', 'RSSI Rx[1]', 'Avg RB count', 'Intermediate KPI']].rolling(rolling_window).sum() / rolling_window

df_rolling = df_rolling.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]', 'Avg RB count'])

def rsrq_to_sinr(RSRQ,x):
	return 1 / ((1 / (12 * RSRQ)) - x)

"""
def rsrp_to_sinr(RSRP, RSSI, N_prb, x):
	S = 12 * N_prb * RSRP
	S_tot = (2/N_prb) * 12 * N_prb * RSRP #where x=RE/RB
	return S / (RSSI - S_tot)
"""
fig, ax1 = plt.subplots()
s10 = df_rolling['SINR Rx[0]']
#s11 = rsrp_to_sinr(df_rolling['RSRQ Rx[0]'],df_rolling['RSSI Rx[0]'],df_rolling['Avg RB count'], 0.01)
s11 = rsrq_to_sinr(df_rolling['RSRQ Rx[0]'], 1/df_rolling['Avg RB count'])
ax1.plot(s10, color='#4C72B0')
ax1.plot(s11, color='#55A868')
ax1.set_xlabel('Measurements')
ax1.set_ylabel('SINR Rx[0] (dB)')
ax1.legend(['SINR Rx[0]', 'RSRQ Rx[0]'], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
ax1.tick_params('y')
#ax1.set_yscale('log', basey=10)


qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')