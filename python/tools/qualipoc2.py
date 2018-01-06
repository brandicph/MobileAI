import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import logging
import sys
import os

from cycler import cycler
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

"""
https://www.laroccasolutions.com/164-rsrq-to-sinr/
"""

script_path = os.path.dirname(os.path.abspath( __file__ ))
# Ostenfeldt Ping
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2018-01-03-00-44-12-0000-5310-7746-0004-S_ping.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping.csv')
# S-TOG B (DTU) Ping
CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
CSV_OUT_FILEPATH = os.path.join(script_path, '../../data/measurement_data.csv')

class QualiPoc(object):

    # Static variables
    LOG_TAG = "QualiPoc"
    LOG_LEVEL = logging.INFO
    LOG_STYLES = {
        'RED': '\x1b[31m',  # red
        'YELLOW': '\x1b[33m',  # yellow
        'GREEN': '\x1b[32m',  # green
        'PINK': '\x1b[35m',  # pink
        'NORMAL': '\x1b[0m',  # normal
        'RESET_SEQ': '\033[0m',
        'BOLD_SEQ': '\033[1m'
    }
    LOG_FORMAT = "{}{}[{}]{}{}[%(asctime)s.%(msecs)03d]{}[%(levelname)s] %(message)s".format(LOG_STYLES['BOLD_SEQ'], LOG_STYLES['RED'], LOG_TAG, LOG_STYLES['RESET_SEQ'], LOG_STYLES['NORMAL'], LOG_STYLES['NORMAL'])

    RSRP_MAPPING = { rssi : i for i, rssi in enumerate(range(-140, -44, 1)) }

    def __init__(self, csv_in_file=None, csv_out_file=None):
        # Setup logger
        self.logger = self.setup_logger()
        self.logger.info('Started parsing...')
        # Set variables
        self.csv_in_file = csv_in_file
        self.csv_out_file = csv_out_file
        # Read QualiPoc csv file
        self.df = pd.read_csv(
            self.csv_in_file,
            parse_dates=["Time"],
            #index_col=['Time'],
            #names=COLUMN_NAMES
        )
        # For tracking
        values_before_parsing = len(self.df)
        # Add special parsing
        # ex. something()

        # For tracking
        values_after_parsing = len(self.df)
        values_ratio_after_parsing = self.safe_division(values_after_parsing, values_before_parsing)
        self.logger.info('Data values: {}/{} ({:.3f})'.format(values_after_parsing, values_before_parsing, values_ratio_after_parsing))

        # Done parsing
        self.logger.info('Done parsing...')

    def setup_logger(self):
        # create logger
        logger = logging.getLogger('console')
        logger.setLevel(self.LOG_LEVEL)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(self.LOG_LEVEL)
        # create formatter
        formatter = ColorFormatter(self.LOG_FORMAT, "%Y-%m-%d %H:%M:%S")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        return logger

    def safe_division(self, a, b):
        return a / b if b else 0.0

    def map_rsrp(self, value):
        return self.RSRP_MAPPING[value]

    def normalize(self, fields, overwrite=False):
        clone = self.df[fields]
        result = clone.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        if overwrite:
            self.df = result
        return result


class ColorFormatter(logging.Formatter):

    LOG_STYLES = {
        'RED': '\x1b[31m',  # red
        'BLUE': '\x1b[34m',  # blue
        'YELLOW': '\x1b[33m',  # yellow
        'GREEN': '\x1b[32m',  # green
        'PINK': '\x1b[35m',  # pink
        'NORMAL': '\x1b[0m',  # normal
        'RESET_SEQ': '\033[0m',
        'BOLD_SEQ': '\033[1m'
    }

    def __init__(self, *args, **kwargs):
        # can't do super(...) here because Formatter is an old school class
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        levelname = record.levelname
        message   = logging.Formatter.format(self, record)
        message   = message.replace("[DEBUG]", "{}[DEBUG]{}".format(self.LOG_STYLES['PINK'], self.LOG_STYLES['NORMAL']))
        message   = message.replace("[INFO]", "{}[INFO]{}".format(self.LOG_STYLES['GREEN'], self.LOG_STYLES['NORMAL']))
        message   = message.replace("[ERROR]", "{}[ERROR]{}".format(self.LOG_STYLES['RED'], self.LOG_STYLES['NORMAL']))
        message   = message.replace("[WARNING]", "{}[WARNING]{}".format(self.LOG_STYLES['YELLOW'], self.LOG_STYLES['NORMAL']))
        return message


def mean_round(x):
    return round(np.mean(x))

def most_common(x):
    (values,counts) = np.unique(x,return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def rolling_mean(data, axis=0):
    return pd.rolling_mean(data, 4, axis=axis).mean(axis=axis)


qp = QualiPoc(CSV_IN_FILEPATH)

print(len(qp.df))
df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])
df = df[df['Intermediate KPI'] > 0]
print(len(df))


mpl.style.use('seaborn')


fig, ax1 = plt.subplots()
s10 = df['SINR Rx[0]']
s11 = df['SINR Rx[1]']
ax1.plot(s10, color='#4C72B0')
ax1.plot(s11, color='#55A868')
ax1.set_xlabel('measurements')
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
df_sinr_rsrp_rolling = df[['SINR Rx[0]', 'RSRP Rx[0]', 'SINR Rx[1]', 'RSRP Rx[1]']].rolling(200, win_type='triang').sum()

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


#plt.title('SINR vs. RSRP\nwithout preprocessing')
#['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
#print(mpl.rcParams['axes.prop_cycle'])

qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')