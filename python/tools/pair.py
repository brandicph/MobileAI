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
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
# S-TOG Ping (Bx)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-04-08-41-51-0000-5310-7746-0004-S_ping2.csv')
# Airport Signal Quality
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_signal_quality.csv')
# Airport Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-16-27-25-0000-5310-7746-0004-S_airport_ping2.csv')
# City Ping
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-17-11-58-0000-5310-7746-0004-S_city_ping2.csv')

# Instatiate data reader
qp = QualiPoc(CSV_IN_FILEPATH)

# Perform filtering
n_samples_before_filtering = len(qp.df)
qp.logger.info("Before filtering: {} samples".format(n_samples_before_filtering))
df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

#df = df[df['Intermediate KPI'] < 200]
#df = df[df['SINR Rx[0]'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))

# Perform rolling mean
rolling_window = 100
df_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]', 'Intermediate KPI']].rolling(rolling_window).sum() / rolling_window

df_rolling = df_rolling.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])

n_samples_after_rolling = len(df_rolling)
qp.logger.info("After rolling: {} samples".format(n_samples_after_rolling))

def rsrq_to_sinr(RSRQ,x):
	return 1 / ((1 / (12 * RSRQ)) - x)

"""
def rsrp_to_sinr(RSRP, RSSI, N_prb, x):
	S = 12 * N_prb * RSRP
	S_tot = (2/N_prb) * 12 * N_prb * RSRP #where x=RE/RB
	return S / (RSSI - S_tot)
"""

g = sns.PairGrid(df_rolling[['SINR Rx[0]', 'RSRP Rx[0]', 'Intermediate KPI']], diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


#sns.pairplot(df_rolling[['SINR Rx[0]', 'RSRP Rx[0]', 'Intermediate KPI']])

#g = sns.jointplot("RSRP Rx[0]", "Intermediate KPI", data=df_rolling, kind="reg", size=7)

qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')