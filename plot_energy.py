import sys

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.title("Energy conservation")
plt.xlabel("Timestep")
plt.ylabel("Energy")

for i in range(len(sys.argv) - 1):
    filename = sys.argv[1 + i]
    theta = "0." + filename.split(".")[0].split("-")[1][1]

    with open(filename) as f:
        lines = f.readlines()

    lines = lines[1:]

    if i == 0:
        energy = np.array([float(l.split(",")[1]) for l in lines])
        plt.plot(energy, linestyle="dashed", label="all-pairs")

    energy = np.array([float(l.split(",")[0]) for l in lines])
    plt.plot(energy, label=r"$\theta = " + theta + "$")

# plt.axhline(0, linestyle="dashed", color="r")

plt.legend()
plt.grid()

plt.show()
