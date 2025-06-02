import sys

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.title("Acceleration error")
plt.xlabel("Timestep")
plt.ylabel(r"$\Delta a / a$")

for i in range(len(sys.argv) - 1):
    filename = sys.argv[1 + i]
    theta = "0." + filename.split(".")[0].split("-")[1][1]

    with open(filename) as f:
        lines = f.readlines()

    lines = lines[1:]

    acc_err = np.array([float(l.split(",")[2]) for l in lines])
    plt.plot(acc_err, label=r"$\theta = " + theta + "$")

plt.legend()
plt.grid()

plt.show()
