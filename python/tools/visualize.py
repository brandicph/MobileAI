import csv_parser
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#sns.set(style="ticks")
sns.set(style="darkgrid", color_codes=True)


data = csv_parser.parse(filepath='../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv', clustered=False)
#data = csv_parser.parse(filepath='../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S.csv', clustered=True)

#data = data[2]

dataframe = pd.DataFrame(
    data=data,
    columns=[
        'timestamp',
        'cycle',
        'band',
        'bandwidth',
        'rssi',
        'rsrp',
        'rsrq',
        'ip_thrpt_dl',
        'bytes_transferred_dl',
        'pdsch_thrpt',
        'cqi0_cqi1',
        'duration'
    ]
)

dataframe_filtered = dataframe[['timestamp', 'rssi', 'rsrp', 'rsrq', 'ip_thrpt_dl', 'bytes_transferred_dl']]

#dataframe[['rssi', 'bytes_transferred_dl']] = dataframe[['rssi', 'bytes_transferred_dl']].rolling(100).sum()

#dataframe[['rssi']] = dataframe[['rssi']].rolling(10).sum()
#dataframe[['bytes_transferred_dl']] = dataframe[['bytes_transferred_dl']].rolling(10).sum()

#rs = np.random.RandomState(11)
#x = rs.gamma(2, size=1000)
#y = -.5 * x + rs.normal(size=1000)

#sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")

#measurements = sns.load_dataset('../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv')
#measurements = pd.read_csv('../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv', header=0, index_col=["Time"], usecols=["Time", "RSSI"], parse_dates=["Time"])
#measurements = pd.read_csv('../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S.csv')

def rolling_mean(data, axis=0):
    return pd.rolling_mean(data, 4, axis=1).mean(axis=axis)


def moving_window(data, n=3):
    return pd.rolling_mean(data, window=n)

#print(dataframe)

#sns.pairplot(dataframe_filtered)
#sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")

#sns.regplot(x="rssi", y="duration", data=dataframe, x_estimator=np.mean)
#sns.jointplot(x="rssi", y="bytes_transferred_dl", data=dataframe, x_estimator=np.mean, kind="reg")
#sns.jointplot(x="rssi", y="bytes_transferred_dl", data=dataframe, kind="reg")
#sns.pairplot(x_vars=["rssi", "rsrp", "rsrq"], y_vars=["duration"], kind="reg", data=dataframe)

sns.pairplot(dataframe_filtered, kind="reg")

#g = sns.jointplot("rssi", "ip_thrpt_dl", data=dataframe, kind="reg", color="r", size=7)
#g = sns.jointplot("rssi", "ip_thrpt_dl", data=dataframe, kind="reg", color="r", size=7)
#g = sns.jointplot("rsrp", "ip_thrpt_dl", data=dataframe, kind="hex", stat_func=kendalltau, color="#4CB391")

plt.show()

#print()