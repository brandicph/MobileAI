import csv
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import logging
import sys
import os

from scipy import interpolate, fft
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from pandas.plotting import andrews_curves

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
from sklearn.datasets import fetch_mldata

from qp import QualiPoc
mpl.style.use('seaborn')

#data = fetch_mldata('mauna-loa-atmospheric-co2').data
#X = data[:, [1]]
#y = data[:, 0]
# #
# Read data
# CSV file path
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
# S-TOG Ping (DTU)
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2018-01-06-08-55-34-0000-5310-7746-0004-S_ping_B.csv')
# Lavensby 1GB download
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-12-06-12-34-57-0000-5310-7746-0004-S_ping2.csv')
# DTU 1GB download
CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_ping2.csv')
#CSV_IN_FILEPATH = os.path.join(SCRIPT_PATH, '../../data/2017-11-27-14-07-44-0000-5310-7746-0004-S_tracking.csv')
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
X_fields = ['RSRP Rx[0]']
#fields = ['Intermediate KPI', 'RSRP', 'RSRQ', 'RSSI', 'Average MCS Index']
#X_fields = ['RSRP','RSRQ', 'RSSI', 'Average MCS Index']
#y_fields = ['Intermediate KPI']
#fields = ['Intermediate KPI', 'RSRP']
#X_fields = ['RSRP']
y_fields = ['Intermediate KPI']
fields = X_fields + y_fields
#df = qp.df.dropna(subset=['Intermediate KPI', 'SINR Rx[0]', 'SINR Rx[1]'])
qp.df = qp.df[:2000].sort_values(by=X_fields)
qp.df = qp.df.dropna(subset=fields)
qp.df = qp.df[qp.df['Intermediate KPI'] > 0]
original = qp.df
df = qp.df
#qp.df = qp.normalize(fields, overwrite=True)

"""
def plotSpectrum(y,Fs):
    #Plots a Single-Sided Amplitude Spectrum of y(t)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]

    plt.plot(frq,abs(Y),'r') # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = qp.df['RSRP Rx[0]'].values

plotSpectrum(y,Fs)
plt.show()
"""

#df = df.groupby('Intermediate KPI').agg({"Intermediate KPI": np.mean, "RSRP Rx[0]": lambda x: x.nunique()})
#df = df.groupby('Time').agg({"Intermediate KPI": np.mean, "RSRP Rx[0]": lambda x: x})
#df = df.groupby('Time').agg({"RSRP Rx[0]": lambda x: x.nunique(),  "Intermediate KPI": lambda x: x})
#df = df[df['SINR Rx[0]'] > 0]
n_samples_after_filtering = len(df)
qp.logger.info("After filtering: {} samples".format(n_samples_after_filtering))


#df = df[::5]

# Perform rolling mean
rolling_window = 2
df_rolling = df[fields].rolling(rolling_window).sum() / rolling_window
df_rolling = df_rolling.dropna(subset=fields)

df_rolling = df_rolling[::rolling_window]
df_rolling = df_rolling.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

# Normal
rolling_window_original = 2
original = original[fields].dropna(subset=fields)
#original = original.sort_values(by=X_fields)
original = original.rolling(rolling_window_original).sum() / rolling_window_original
bla = original
original = original.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
original = original[::]

#df_rolling = df[fields].rolling(rolling_window).sum() / rolling_window
#df_rolling = df_rolling.dropna(subset=fields)





n_samples_after_rolling = len(df_rolling)
qp.logger.info("After rolling: {} samples".format(n_samples_after_rolling))


X = df_rolling[X_fields].values
y = df_rolling[y_fields].values.ravel()

# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=True, n_restarts_optimizer=2)
#gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, normalize_y=True)

#kernel_gpml = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp = GaussianProcessRegressor(kernel=kernel_gpml, n_restarts_optimizer=9)

gp.fit(X, y)


# 5.4 Model Selection for GP Regression (p112 - p124)
# http://www.gaussianprocess.org/gpml/chapters/RW.pdf
print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

# Kernel with optimized parameters
k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() * 1.10, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)


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

# Illustration
plt.scatter(X, y, c='black', marker='+', label=u'Observations', alpha=0.5)
plt.plot(original['RSRP Rx[0]'], original['Intermediate KPI'], color='black', alpha=0.2, label=u'Original')
plt.plot(X_, y_pred, c='#C44E52', label=u'Prediction')
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.2, color='#C44E52', label='Confidence interval')#color='#C44E52')
plt.xlim(X_.min(), X_.max())
plt.xlabel("RSRP")
plt.ylabel("Throughput")
plt.title("Gaussian Process (GPML)\n- normalized -")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()