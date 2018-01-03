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

"""
https://www.laroccasolutions.com/164-rsrq-to-sinr/
"""

script_path = os.path.dirname(os.path.abspath( __file__ ))
# Ostenfeldt Ping
CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2018-01-03-00-44-12-0000-5310-7746-0004-S_ping.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping.csv')
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

#qp.df = qp.normalize(['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'Test KPI avg', 'SINR Rx[0]'], overwrite=True)

plot_data = qp.df[qp.df["Intermediate KPI"] > 0]

#print(plot_data)

#plot_data = plot_data[['Cycles', 'RSSI', 'RSRP', 'SINR Rx[0]', 'SINR Rx[1]', 'Bytes Transferred', 'Bitrate']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

qp.logger.info('Started plotting...')

sns.set(style="darkgrid", color_codes=True)

plot_data = plot_data.dropna(axis=1, how='any')

plot_data_reduced = plot_data.drop_duplicates(subset='Time', keep='last')
#plot_data_reduced['Intermediate KPI'] = plot_data_reduced['Intermediate KPI'].rolling(window=5).mean()

#sns.barplot(data = plot_data_reduced.reset_index(), x = "Time", y = "RSSI")

#g = sns.jointplot(plot_data_reduced['RSRP'], plot_data_reduced['Intermediate KPI'], kind="kde", size=7, space=0)

# PAIR
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'Test KPI avg']], kind="reg")#, dropna=True)
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'SINR Rx[0]']], diag_kind="kde", markers="+", diag_kws=dict(shade=True), plot_kws=dict(s=10, edgecolor="b", linewidth=1))#, dropna=True)
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'Intermediate KPI']], hue='Intermediate KPI', kind="reg")#, dropna=True)

#sns.jointplot(x="RSRP", y="Intermediate KPI", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)
#sns.jointplot(y="RSRP", x="Intermediate KPI", data=plot_data_reduced, x_estimator=np.mean, kind="reg", color="r", size=7)

plot_data_reduced.set_index('Time')
#plot_data_rolling = plot_data_reduced[['RSSI', 'RSRP', 'Intermediate KPI', 'Test KPI avg']].rolling(2, win_type='triang').sum()
plot_data_rolling = plot_data_reduced[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'SINR Rx[0]']].rolling(2, win_type='triang').sum()
#sns.jointplot(x="RSRP", y="Intermediate KPI", data=plot_data_reduced, x_estimator=np.mean, kind="reg", color="r", size=7)
#sns.jointplot(x="RSRP", y="Intermediate KPI", data=plot_data_rolling, kind="reg", color="r", size=7)

#sns.pairplot(plot_data_reduced[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]']], kind="reg")#, dropna=True)



"""
# TEST
plot_data_copy = plot_data_reduced

fig = plt.figure()
ax = Axes3D(fig)
#plot_data_copy = plot_data_copy.interpolate(method='cubic')

plot_data_copy_normalized = plot_data_copy[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'SINR Rx[0]']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
plot_data_normalized = plot_data_copy[['RSSI', 'RSRP', 'RSRQ', 'Intermediate KPI', 'SINR Rx[0]']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

# https://stackoverflow.com/questions/35157650/smooth-surface-plot-with-pyplot
xnew, ynew = np.mgrid[-1:1:80j, -1:1:80j]
tck = interpolate.bisplrep(plot_data_copy_normalized['RSRP'], plot_data_copy_normalized['SINR Rx[0]'], plot_data_copy_normalized['Intermediate KPI'])
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

#surf = ax.plot_surface(plot_data_copy['RSSI'], plot_data_copy['Bitrate'], plot_data_copy['SINR Rx[0]'], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.plot_trisurf(plot_data_copy['RSSI'], plot_data_copy['SINR Rx[0]'], plot_data_copy['Bitrate'], cmap=cm.jet, linewidth=0.2)

surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.jet, rstride=1, cstride=1, alpha=None, antialiased=False)
ax.set_xlabel('RSRṔ')
ax.set_ylabel('SINR Rx[0]')
ax.set_zlabel('Intermediate KPI')

cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
#cbar.solids.set_antialiased(True)
#cbar.solids.set(alpha=1)
plt.title('Normalized 3D Mesh')
"""

plot_data_reduced['100MA1'] = pd.rolling_mean(plot_data_reduced['SINR Rx[0]'], 100)
plot_data_reduced['100MA2'] = pd.rolling_mean(plot_data_reduced['RSRP'], 100)
plot_data_reduced['100MA3'] = pd.rolling_mean(plot_data_reduced['Intermediate KPI'], 100)

threedee = plt.figure().gca(projection='3d')
threedee.scatter(plot_data_reduced["100MA1"], plot_data_reduced["100MA2"], plot_data_reduced["100MA3"])
threedee.set_xlabel('SINR Rx[0]')
threedee.set_ylabel('RSRP')
threedee.set_zlabel('Intermediate KPI')
plt.show()


def f(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


x = np.linspace(plot_data_reduced['100MA1'].min(), plot_data_reduced['100MA1'].max(), 30)
y = np.linspace(plot_data_reduced['100MA2'].min(), plot_data_reduced['100MA2'].max(), 30)
z = np.linspace(plot_data_reduced['100MA3'].min(), plot_data_reduced['100MA3'].max(), 30)

X, Y = np.meshgrid(x, y)

Z = f(X, Y, z)

fig = plt.figure()
ax = Axes3D(fig)
#ax.plot3D(plot_data_reduced["100MA1"], plot_data_reduced["100MA2"], plot_data_reduced["100MA3"], 'gray')
#ax.scatter3D(plot_data_reduced["100MA1"], plot_data_reduced["100MA2"], plot_data_reduced["100MA3"], c=plot_data_reduced["100MA3"], cmap='viridis');
ax.plot_wireframe(X, Y, Z, color='black')
#plt.tricontour(X.ravel(), Y.ravel(), Z.ravel()) 


qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')