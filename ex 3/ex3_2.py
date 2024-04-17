"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""
"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""

import numpy as np
from scipy.ndimage import convolve
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import time

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load data
X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")

# Generate augmented data
X, Y = nudge_dataset(X, y)

# Scale features
X = minmax_scale(X, feature_range=(0, 1))  

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Define models
logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# Define dimensions for n_components
dimensions = [i**2 for i in range(2, 21)]

# Initialize lists to store results
average_precision_values = []
training_times = []

# Loop over dimensions
for dim in dimensions:
    # Set RBM n_components
    rbm.n_components = dim
    
    # Set RBM hyperparameters
    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    logistic.C = 6000
    
    # Measure training time
    start_time = time.perf_counter()
    
    # Fit RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, Y_train)
    
    # Measure training time
    end_time = time.perf_counter()
    training_time = end_time - start_time
    training_times.append(training_time)
    
    # Make predictions
    Y_pred = rbm_features_classifier.predict(X_test)
    
    # Calculate average precision
    avg_precision = metrics.precision_score(Y_test, Y_pred, average='weighted')
    average_precision_values.append(avg_precision)

    # Plot RBM components
    plt.figure(figsize=(10, 10))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(int(np.sqrt(dim)), int(np.sqrt(dim)), i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()


# Plot average precision against number of components
plt.figure(figsize=(8, 6))
plt.plot(dimensions, average_precision_values, marker='o', linestyle='-')
plt.axhline(y=0.78, color='r', linestyle='--', label='Logistic Regression on Raw Pixels (0.78)')
plt.title('Average Precision vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Average Precision (macro avg)')
plt.legend()
plt.grid(True)
plt.show()

# Plot time per run against number of components
plt.figure(figsize=(8, 6))
plt.plot(dimensions, training_times, marker='o', linestyle='-')
plt.title('Training Time per Run vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Training Time (seconds)')
plt.grid(True)
plt.show()