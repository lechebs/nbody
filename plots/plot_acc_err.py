import sys

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

#plt.title(r"Average step-local acceleration error for different values of $\theta$")
plt.xlabel("Timestep")
plt.ylabel(r"$\Delta a / a$")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_mapping = [2, 1, 3]

plt.plot([])

for i in range(len(sys.argv) - 1):
    filename = sys.argv[1 + i]
    theta = "0." + filename.split(".")[0].split("-")[-1][1]

    with open(filename) as f:
        lines = f.readlines()

    lines = lines[2:1001]

    acc_err = np.array([float(l.split(",")[2]) for l in lines])
    plt.plot(acc_err, label=r"$\theta = " + theta + "$",
             color=colors[color_mapping[i]])


plt.legend()
plt.grid()

plt.show()
