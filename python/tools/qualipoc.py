import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import logging
import sys
import os

from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

"""
https://www.laroccasolutions.com/164-rsrq-to-sinr/
"""

script_path = os.path.dirname(os.path.abspath( __file__ ))
# DTU 10MB / 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv')
# DTU 1GB download
#CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S.csv')
# Lavensby 1GB download
CSV_IN_FILEPATH = os.path.join(script_path, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv')
CSV_OUT_FILEPATH = os.path.join(script_path, '../../data/measurement_data.csv')

COLUMN_NAMES = {
    "Time", #0
    "Network", #1
    "Operator", #2
    "Technology", #3
    "Data technology", #4
    "Cell Id",
    "DL EARFCN",
    "RSSI", #7
    "PCI",
    "RSRP", #9
    "RSRQ", #10
    "TxPower",
    "MCC",
    "MNC",
    "MCC/MNC",
    "LAC",
    "LAC/CID",
    "BTS Cell Name",
    "BTS Cell Name 2",
    "BTS Cell Name 3",
    "----------",
    "IP address",
    "Last started test",
    "Job name",
    "Cycles", #24
    "Test KPI type",
    "Test KPI avg",
    "Test KPI worst",
    "Test KPI best",
    "Test KPI last",
    "Intermediate KPI",
    "IP Thrpt DL", #31
    "IP Thrpt UL",
    "----------",
    "RSRP",
    "RSRQ",
    "RSSI",
    "SINR Rx[0]", # 37 CHECK
    "SINR Rx[1]", # 38 CHECK
    "RSRP Rx[0]", # 39
    "RSRQ Rx[0]", # 40 CHECK
    "RSSI Rx[0]", # 41
    "RSRP Rx[1]", # 42
    "RSRQ Rx[1]", # 43 CHECK
    "RSSI Rx[1]", # 44
    "PCI", #45
    "DL EARFCN", #46
    "Bandwidth", #47
    "TAC",
    "Tx Antennas",
    "QRxLevMin",
    "Pmax",
    "MaxTxPower",
    "SRxLev",
    "SIntraSearch",
    "SNonIntraSearch",
    "Intra PCI",
    "Inter Freq./ PCI",
    "PLMNID (MCC/MNC)",
    "eNB/Sector ID",
    "eNB",
    "RF Band (LTE)", #61
    "CP distribution",
    "Timing advance",
    "EMM State",
    "QCI", #65
    "Max bit rate UL",
    "Max bit rate DL",
    "Guaranteed bit rate UL",
    "Guaranteed bit rate DL",
    "----------",
    "Num of Subframes",
    "Max Num of Layers",
    "Avg RB count",
    "Bytes Transferred", #74
    "Sched Thrpt",
    "PDSCH Thrpt", #76
    "BLER",
    "Average TB Size",
    "Num of TBs",
    "Num Pass 1st Attempt",
    "Average MCS Index",
    "Transmission Mode",
    "Num QPSK (LTE DL)",
    "Num 16QAM (LTE DL)",
    "Num 64QAM (LTE DL)",
    "Number of carriers DL",
    "Num 256QAM (LTE DL)",
    "Num ACKs",
    "Num NACKs",
    "Reporting Mode",
    "CQI0/CQI1", #91
    "PUCCH Tx power",
    "Bytes Transferred",
    "Sched Thrpt",
    "PUSCH Thrpt",
    "Retransmission Rate",
    "Number of TBs",
    "TxPower",
    "Num ACKs",
    "Num NACKs",
}

#USE_COLS = [0,1,2,3,4,7,9,10,24,31,45,46,47,61,65,74,76,91]

USE_COLS = [0,1,2,3,4,7,9,10,24,31,37,38,39,40,41,42,43,44,45,46,47,61,65,74,76,91]

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
            converters={24: self.parse_cycles, 91: self.parse_cqi },
            usecols=USE_COLS,
            #names=COLUMN_NAMES
        )
        # For tracking
        values_before_parsing = len(self.df)
        # Drop rows that include NaN value in specific columns
        self.df = self.df.dropna(subset=['Cycles','SINR Rx[0]', 'SINR Rx[1]', 'RSSI', 'RSRP', 'RSRQ', 'Bytes Transferred'])
        # Ensure specific technology
        self.df = self.df[self.df["Data technology"] == 'LTE']
        # For tracking
        values_after_parsing = len(self.df)
        values_ratio_after_parsing = self.safe_division(values_after_parsing, values_before_parsing)
        self.logger.info('Data values: {}/{} ({:.3f})'.format(values_after_parsing, values_before_parsing, values_ratio_after_parsing))
        # Calculate commulative transfer rate
        # DataFrame.cumsum() will not do any good
        self.df = self.calculate_transfer_rate(self.df)

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

    def parse_cqi(self, data):
        arr = data.replace(' ', '').split('/')
        if len(arr) <= 1:
            return np.nan
        for i,v in enumerate(arr):
            arr[i] = float(v)
        return arr

    def parse_cycles(self, data):
        # Parse cycles "x/n" (ex. 2/5)
        arr = data.replace(' ', '').split('/')
        if len(arr) <= 1:
            return np.nan
        for i,v in enumerate(arr):
            arr[i] = int(v)
        return arr[0]

    def safe_division(self, a, b):
        return a / b if b else 0.0

    def map_rsrp(self, value):
        return self.RSRP_MAPPING[value]

    def normalize(self, fields):
        return self.df[fields].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    def calculate_transfer_rate(self, data):
        # Add missing fields with zero padding
        data['Bytes Transferred Cycle'] = 0.0
        data['Duration'] = 0.0
        data['Duration Seconds'] = 0.0
        data['Bitrate'] = 0.0
        data['Bytes Transferred Interval'] = 0.0
        data['RSRP Mapping'] = 0.0

        # Local values
        prev_value = 0
        prev_cycle = 0
        total = 0
        start_time = None

        # Loop through DataFrame using indexes
        # OBS. Index lookup shows to be much faster than data.iterows()
        for i in data.index.values:
            # Set the starting time
            if start_time == None: 
                start_time = data.at[i, "Time"]

            # Detect cycles
            if data.at[i, 'Cycles'] != prev_cycle:
                start_time = data.at[i, "Time"]
                prev_value = 0
                total = 0

            if data.at[i, "Bytes Transferred"] != prev_value:
                # If the the current bytes transferred is different, then add to total
                # This is to prevent duplicate values found in the QualiPoc dataset
                total += prev_value
                # Register NaN and then check
                data.at[i, 'Bytes Transferred Interval'] = prev_value

            # Set the total amount of transferred bytes to current total
            data.at[i, 'Bytes Transferred Cycle'] = total

            # Map RSRP
            data.at[i, 'RSRP Mapping'] = self.RSRP_MAPPING[np.floor(data.at[i, 'RSRP'])]

            # Calculate total duration (deltatime)
            data.at[i, 'Duration'] = data.at[i, "Time"] - start_time
            
            # Calculate total cycle duration in seconds
            duration_seconds = data.at[i, 'Duration'].total_seconds() #self.df.at[i, 'Duration'].seconds + self.df.at[i, 'Duration'].microseconds/1E6
            data.at[i, 'Duration Seconds'] = duration_seconds
            
            # Calulate bitrate mbits/sec
            transferred_cycle_bits = total * 8.0
            transferred_cycle_mbits = transferred_cycle_bits / 1E6
            data.at[i, 'Bitrate'] = self.safe_division(transferred_cycle_mbits, duration_seconds)

            # Set the current bytes transferred as the previus value
            prev_value = data.at[i, "Bytes Transferred"]
            prev_cycle = data.at[i, 'Cycles']
            
            # Print values for debugging purpose
            self.logger.debug("{} / {} = {}".format(transferred_cycle_mbits, duration_seconds, data.at[i, 'Bitrate']))

        return data

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


qp = QualiPoc(CSV_IN_FILEPATH)

#qp.df = qp.normalize(['Cycles', 'RSSI', 'RSRP', 'SINR Rx[0]', 'SINR Rx[1]', 'Bytes Transferred', 'Bitrate'])

plot_data = qp.df[qp.df["Cycles"] >= 0]
plot_data = plot_data[plot_data["Bitrate"] <= 150]
plot_data = plot_data[plot_data["Bitrate"] > 0]
plot_data = plot_data[plot_data["Bytes Transferred"] > 0]
#print(plot_data)

#plot_data = plot_data[['Cycles', 'RSSI', 'RSRP', 'SINR Rx[0]', 'SINR Rx[1]', 'Bytes Transferred', 'Bitrate']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

qp.logger.info('Started plotting...')

sns.set(style="darkgrid", color_codes=True)

plot_data = plot_data.dropna(axis=1, how='any')


# PAIR
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'Bytes Transferred', 'Bitrate', 'RSRP Mapping']], kind="reg")#, dropna=True)
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'Bytes Transferred', 'Bitrate', 'RSRP Mapping']], diag_kind="kde", markers="+", diag_kws=dict(shade=True), plot_kws=dict(s=10, edgecolor="b", linewidth=1))#, dropna=True)
#sns.pairplot(plot_data[['RSSI', 'RSRP', 'Bytes Transferred', 'Bitrate', 'RSRP Mapping']], hue='RSRP Mapping', kind="reg")#, dropna=True)

#sns.pairplot(plot_data[['RSSI', 'RSRP','SINR Rx[0]', 'SINR Rx[1]', 'Bytes Transferred', 'Bitrate']], kind="reg")#, dropna=True)

# HEX
#sns.jointplot(x="RSRP Mapping", y="Bitrate", data=plot_data, kind="hex", stat_func=kendalltau, color="#4CB391", size=7)
# REG
#sns.jointplot(x="RSRP Mapping", y="Bitrate", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)
#sns.jointplot(x="RSRP Mapping", y="Bytes Transferred", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)
#sns.jointplot(x="RSRP Mapping", y="Bytes Transferred", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)

#sns.jointplot(x="SINR Rx[0]", y="Bytes Transferred", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)
#sns.jointplot(x="SINR Rx[1]", y="Bytes Transferred", data=plot_data, x_estimator=np.mean, kind="reg", color="r", size=7)

# BOX
#sns.boxplot(data=plot_data[['RSSI', 'RSRP','SINR Rx[0]', 'SINR Rx[1]', 'Bytes Transferred', 'Bitrate']])
#sns.boxplot(data=plot_data[['Bytes Transferred']])

def mean_round(x):
    return round(np.mean(x))

def most_common(x):
    (values,counts) = np.unique(x,return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

# TEST
plot_data_copy = plot_data
#f = {'Bytes Transferred': 'sum', 'Bitrate': mean_round, 'Cycles': mean_round, 'RSRP Mapping': mean_round, 'SINR Rx[0]': mean_round, 'SINR Rx[1]': mean_round}
f = {'Bytes Transferred': 'sum', 'Bitrate': most_common, 'Cycles': most_common, 'RSRP Mapping': most_common, 'RSRQ': most_common, 'SINR Rx[0]': most_common, 'SINR Rx[1]': most_common}
g = pd.TimeGrouper(freq='1Min')
plot_data_copy = plot_data_copy.set_index('Time').groupby(g).agg(f)
print(plot_data_copy)
plot_data_copy["Timestamp"] = plot_data_copy.index

#print(plot_data['Cycles'])
#sns.tsplot(data=plot_data_copy, time="Timestamp", unit="Bitrate", condition="Cycles", value="Bitrate")
#sns.factorplot(y="Bytes Transferred", x="Bitrate", hue="RSRP Mapping", data=plot_data_copy, size=6, kind="bar", palette="muted")
sns.jointplot(x="SINR Rx[0]", y="Bytes Transferred", data=plot_data_copy, x_estimator=np.mean, kind="reg", color="r", size=7)

# TIME SERIES
#sns.barplot(x="Time", y="Bytes Transferred", data=plot_data, palette="Set3")
#plot_data_deduplicate = plot_data[plot_data["Bytes Transferred Interval"] > 0]
#sns.tsplot(data=plot_data_deduplicate['Bytes Transferred Interval'], time=plot_data_deduplicate['Time'])
# BAR

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
#https://stackoverflow.com/questions/35168016/unable-to-plot-dataframe-using-seaborn-barplot


#plot_data.set_index('Time', inplace=True)
#plot_data = plot_data.resample("1Min").sum()

plot_data_copy['Timestamp'] = [x.to_pydatetime().strftime("%H:%M:%S") for x in plot_data_copy.index]
#print(plot_data['Timestamp'])

sns.barplot(x=plot_data_copy['Timestamp'], y=plot_data_copy["Bytes Transferred"], color="r", ax=ax1)
sns.barplot(x=plot_data_copy['Timestamp'], y=plot_data_copy["Bitrate"], color="g", ax=ax2)
sns.barplot(x=plot_data_copy['Timestamp'], y=plot_data_copy["SINR Rx[0]"], color="b", ax=ax3)
sns.barplot(x=plot_data_copy['Timestamp'], y=plot_data_copy["RSRQ"], color="y", ax=ax4)


"""
def myFormatter(x, pos):
    return x+1

"ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter))
ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter))
"""

"""
date_fmt = '%H:%M:%S'

formatter = mdates.DateFormatter(date_fmt)
ax2.plot_date(plot_data['Timestamp'], plot_data["Bitrate"])
ax2.xaxis.set_major_formatter(formatter)
"""

#sns.tsplot(time="Time", unit="Bytes Transferred", data=[plot_data[["Time", "Bytes Transferred"]]])
#sns.factorplot(y="Bytes Transferred", data=plot_data, palette="BuPu", size=6, aspect=1.5)


#sns.jointplot(x="Bitrate", y="Bytes Transferred", data=plot_data, kind="reg", color="r", size=7)

# Add new parameters
# Implement SVM - kernel
# Gauss process

qp.logger.info('Showing plot...')

plt.show()

qp.logger.info('Exit...')