from dataUncert import *
import matplotlib.pyplot as plt

x = variable([3, 4, 5, 6], 'm', [0.15, 0.3, 0.45, 0.6])
y = variable([10, 20, 30, 40], 'C', [2, 3, 4, 5])

F = lin_fit(x, y)


fig, ax = plt.subplots()
F.scatter(ax)
F.plot(ax)
ax.set_xlabel('Distance')
ax.set_ylabel('Temperature')
F.addUnitToLabels(ax)
ax.legend()
fig.tight_layout()
plt.show()
