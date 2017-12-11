import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt

#CSV_IN_FILEPATH = '../../data/2017-11-17-12-39-33-0000-5310-7746-0004-S.csv'
#CSV_IN_FILEPATH = '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S.csv'
CSV_IN_FILEPATH = '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv'
CSV_OUT_FILEPATH = '../../data/measurement_data.csv'

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
    "SINR Rx[0]",
    "SINR Rx[1]",
    "RSRP Rx[0]",
    "RSRQ Rx[0]",
    "RSSI Rx[0]",
    "RSRP Rx[1]",
    "RSRQ Rx[1]",
    "RSSI Rx[1]",
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

USE_COLS = [0,1,2,3,4,7,9,10,24,31,45,46,47,61,65,74,76,91]

class QualiPoc(object):

    def __init__(self, csv_in_file=None, csv_out_file=None):
        self.csv_in_file = csv_in_file
        self.csv_out_file = csv_out_file
        self.df = pd.read_csv(
            self.csv_in_file,
            parse_dates=["Time"],
            converters={24: self.parse_cycles, 91: self.parse_cqi },
            usecols=USE_COLS,
            #names=COLUMN_NAMES
        ).dropna(subset=['Cycles', 'RSSI', 'RSRP', 'RSRQ', 'Bytes Transferred'])

        self.df = self.df[self.df["Data technology"] == 'LTE']
        #self.df['Bitrate'] = self.df["Bytes Transferred"].cumsum()
        self.df = self.calculate_transfer_rate(self.df)

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

    def calculate_transfer_rate(self, data):
        # Add missing fields with zero padding
        data['Bytes Transferred Cycle'] = 0.0
        data['Duration'] = 0.0
        data['Duration Seconds'] = 0.0
        data['Bitrate'] = 0.0

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

            # Set the total amount of transferred bytes to current total
            data.at[i, 'Bytes Transferred Cycle'] = total

            # Calculate total duration (deltatime)
            data.at[i, 'Duration'] = data.at[i, "Time"] - start_time
            
            # Calculate total cycle duration in seconds
            duration_seconds = data.at[i, 'Duration'].total_seconds() #self.df.at[i, 'Duration'].seconds + self.df.at[i, 'Duration'].microseconds/1E6
            data.at[i, 'Duration Seconds'] = duration_seconds
            
            # Calulate bitrate bits/sec
            transferred_cycle_bits = total * 8.0
            transferred_cycle_mbits = transferred_cycle_bits / 1E6
            data.at[i, 'Bitrate'] = (transferred_cycle_mbits / duration_seconds) if duration_seconds > 0 else 0.0

            # Set the current bytes transferred as the previus value
            prev_value = data.at[i, "Bytes Transferred"]
            prev_cycle = data.at[i, 'Cycles']
            
            # Print values for debugging purpose
            #print("{} / {} = {}".format(transferred_cycle_mbits, duration_seconds, data.at[i, 'Bitrate']))

        return data


qp = QualiPoc(CSV_IN_FILEPATH)

plot_data = qp.df[qp.df["Cycles"] == 2.0]
#print(plot_data)


sns.regplot(x="RSSI", y="Bitrate", data=plot_data)

plt.show()