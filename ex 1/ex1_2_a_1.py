"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x1 = np.random.normal(0,1,1000)
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

plt.scatter(x1, x2, s = 5)
plt.axis('square')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.rc('grid', linestyle='-', color='white', linewidth=0.5)
plt.grid(True)
plt.gca().set_facecolor('lightgrey')
plt.show()


plt.hist(x1, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
mean_x1, std_x1 = np.mean(x1), np.std(x1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_x1, std_x1)
plt.plot(x, p, 'red', linewidth=2)
plt.xlim(-5,5)
plt.ylim(0,0.42)
plt.grid(True)
plt.gca().set_facecolor('lightgrey')
plt.show()

plt.hist(x2, bins=30,orientation='horizontal', density=True, alpha=0.7,
         color='blue',edgecolor='black')
mean_x2, std_x2 = np.mean(x2), np.std(x2)
ymin, ymax = plt.ylim()
y = np.linspace(ymin, ymax, 100)
p2 = norm.pdf(y, mean_x2, std_x2)
plt.plot(p2, y, 'red', linewidth=2)
plt.ylim(-5,5)
plt.xlim(0,0.42)
plt.grid(True)
plt.gca().set_facecolor('lightgrey')
plt.show()

# Explanation of Normalization:
"""
The normalization process involves transforming x1 values to x2 based on a 
defined condition.
For each value in x1:
- If the absolute value of x1 is less than or equal to 1, x2 takes the same
 value as x1.
- If the absolute value of x1 is greater than 1, x2 takes the negative of x1.
"""