import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Marian Kravec

data = []
data_ind = []

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

with open("data.txt") as f:
    for i in range(5):
        f.readline()
    for i in range(500):
        r = f.readline().strip().split()
        data.append(float(r[1]))
        data_ind.append(int(r[0]))

data = np.array(data)
data_ind = np.array(data_ind)

plt.plot(data_ind, data, label="Original")

n = 7

plt.plot(data_ind[:-(n-1)], moving_average(data, n),  label="Moving Average")

plt.plot(data_ind[(n-1)//2:-((n-1)//2)], moving_average(data, n), label="Centered Moving Average")

plt.plot(data_ind[(n-1):-(n-1)], moving_average(moving_average(data, n), n), label="Double Centered Moving Average")

reg = LinearRegression().fit(data_ind.reshape(-1, 1), data)

plt.plot(data_ind, reg.predict(data_ind.reshape(-1, 1)), label="Linear regression")

plt.legend()


fig, ax = plt.subplots(2,2)

ax[0, 0].plot(data_ind[:-(n-1)], data[:-(n-1)],  label="Original")
ax[0, 0].plot(data_ind[:-(n-1)], data[:-(n-1)]-moving_average(data, n),  label="minus Moving Average")
ax[0, 0].set_title('Mov. Avg. norm, window='+str(n))

ax[0, 1].plot(data_ind[(n-1)//2:-((n-1)//2)], data[(n-1)//2:-((n-1)//2)],  label="Original")
ax[0, 1].plot(data_ind[(n-1)//2:-((n-1)//2)], data[(n-1)//2:-((n-1)//2)]-moving_average(data, n), label="minus Centered Moving Average")
ax[0, 1].set_title('Centered Mov. Avg. norm, window='+str(n))

ax[1, 0].plot(data_ind[(n-1):-(n-1)], data[(n-1):-(n-1)],  label="Original")
ax[1, 0].plot(data_ind[(n-1):-(n-1)], data[(n-1):-(n-1)]-moving_average(moving_average(data, n), n), label="minus Double Centered Moving Average")
ax[1, 0].set_title('Double Centered Mov. Avg. norm, window='+str(n))

ax[1, 1].plot(data_ind, data,  label="Original")
ax[1, 1].plot(data_ind, data-reg.predict(data_ind.reshape(-1, 1)), label="Linear regression")
ax[1, 1].set_title('Linear regression')

fig2, ax2 = plt.subplots(2)

simpExp = SimpleExpSmoothing(data, initialization_method="heuristic").fit(
    smoothing_level=0.2, optimized=False
)

doubleExp = Holt(data, initialization_method="estimated").fit(
    smoothing_level=0.1, smoothing_trend=0.1, optimized=False
)



k = 50

ax2[0].plot(data_ind, data, label="Original")

ax2[0].plot(list(range(501, 501+k)), simpExp.forecast(k), label="Single Exponential Forecast (smoothing level = 0.2)")

ax2[0].legend()

ax2[1].plot(data_ind, data, label="Original")

ax2[1].plot(list(range(501, 501+k)), doubleExp.forecast(k), label="Double Exponential Forecast (smoothing level = 0.1, smoothing trend = 0.1)")

ax2[1].legend()


plt.show()










