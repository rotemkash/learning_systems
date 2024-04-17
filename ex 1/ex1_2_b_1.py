"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x1 = np.linspace(-4, 4, 100)

def transform_x2(x):
    """
    create x2 by the definition: X2 = {X1 if |X1|<=1; -X1 if |X1| > 1}

    Parameters
    ----------
    x : an integer of type float

    Returns
    -------
    same integer +/- by definition
    """
    if abs(x) <= 1:
        return x
    else:
        return -x

x2 = np.array([transform_x2(x) for x in x1])

x1_values, x2_values = np.meshgrid(x1, x2)

pdf_x1 = norm.pdf(x1, np.mean(x1), np.std(x1))
pdf_x2 = norm.pdf(x2, np.mean(x2), np.std(x2))

joint_pdf = np.matmul(pdf_x1.reshape(100, 1), pdf_x2.reshape(1, 100))

fig,ax = plt.subplots(subplot_kw={"projection":"3d"})

ax.plot_surface(x1_values, x2_values, joint_pdf, cmap='viridis')

ax.contourf(x1_values, x2_values, joint_pdf, zdir='z', offset=-0.2, 
            cmap='viridis')


ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-0.2, 0.2])  
ax.set_zticks(np.arange(0, 0.21, 0.05))

plt.show()

