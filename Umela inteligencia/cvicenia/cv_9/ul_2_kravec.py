import matplotlib.pyplot as plt
import numpy as np


data = []
with open("data.txt", "r") as f:
    for line in f:
        data.append(float(line.strip()))

data = np.array(data)

def err_fun(x, y):
    return (x-y)@(x-y)

def gen_time_ser(mi, y_0,length=500):
    ys = [y_0]
    for i in range(length):
        ys.append(mi*ys[-1]*(1-ys[-1]))

    return np.array(ys[1:])

min_mi = 0
max_mi = 4
min_y0 = 0
max_y0 = 1

grid = 10

for i in range(5):
    step_mi = (max_mi-min_mi)/grid
    step_y0 = (max_y0-min_y0)/grid
    best_mi = -1
    best_y0 = -1
    best_err = float("inf")
    best_ts = None
    for mi_coef in range(grid):
        for y_coef in range(grid):
            mi = min_mi + mi_coef*step_mi
            y0 = min_y0 + y_coef*step_y0
            ts = gen_time_ser(mi, y0)
            err = err_fun(data, ts)
            if err < best_err:
                best_mi = mi
                best_y0 = y0
                best_err =err
                best_ts = ts
    min_mi = best_mi - (step_mi/2)
    max_mi = best_mi + (step_mi/2)
    min_y0 = best_y0 - (step_y0/2)
    max_y0 = best_y0 + (step_y0/2)

print("mi:",best_mi)
print("y0:",best_y0)
print("SSE:", best_err) 

fig, ax = plt.subplots(2, 1)

prvych_n = 500

ax[0].plot(data[:prvych_n], label="Original")
ax[0].plot(best_ts[:prvych_n], label="Časový rad")
ax[0].legend(loc="upper right")

ax[1].plot((data-best_ts)[:prvych_n], label="Diferencie")
ax[1].legend(loc="upper right")

plt.show()
    
