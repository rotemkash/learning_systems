"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""
import numpy as np
import matplotlib.pyplot as plt

signal = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
kernel = np.array([0, 1, -1, 0])
result = np.convolve(signal, kernel, mode='same')

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

axs[0].step(np.arange(len(signal)), signal, color='blue', where='mid')
axs[0].text(5.5, 0.8, 'signal', ha='center', va='center')
axs[0].set_ylim(-1.1, 1.1)
axs[0].set_xticks([])

axs[1].step(np.arange(len(kernel)), kernel, color='green', where='mid')
axs[1].text(5.0, 0.8, 'kernel', ha='center', va='center')
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_xticks([])

axs[2].step(np.arange(len(result)), result, color='red', where='mid')
axs[2].text(5.1, 0.8, 'result', ha='center', va='center')
axs[2].set_ylim(-1.1, 1.1)
axs[2].set_xlim(-0.4, 11.5)
axs[2].set_xticks(np.arange(0, 11,2))

plt.show()