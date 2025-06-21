import sys

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

#plt.title(r"Energy conservation of a self-gravitating disk of 32768 particles "
#          r"for different values of $\theta$")
plt.xlabel("Timestep")
plt.ylabel("Total Energy")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_mapping = [2, 1, 3]

name = ["double", "float"]

for i in range(len(sys.argv) - 1):
    filename = sys.argv[1 + i]
    theta = "0." + filename.split(".")[0].split("-")[-1][1]

    with open(filename) as f:
        lines = f.readlines()

    lines = lines[2:5001]

    if i == 0:
        energy = np.array([float(l.split(",")[1]) for l in lines])
        plt.plot(energy, linestyle="dashed", label="all-pairs (double)")

    energy = np.array([float(l.split(",")[0]) for l in lines])
    plt.plot(energy, label=r"$\theta = " + theta + "$ (" + name[i] + ")",
             color=colors[color_mapping[i]])

# plt.axhline(0, linestyle="dashed", color="r")

plt.legend()
plt.grid()

plt.show()
