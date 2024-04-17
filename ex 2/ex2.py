"""
Written and submitted by Rotem Kashani 209073352 and David Koplev 208870279
"""

"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

"""
Functions
"""
def sum_of_values(image_matrix):
    """
    Calculate the sum of all values in the image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input 8x8 image matrix

    Returns
    -------
    float
        Sum of all values in the matrix
    """
    return np.sum(image_matrix)


def horizontal_symmetry_index(image_matrix):
    """
    Calculate the horizontal symmetry index of the image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input image matrix

    Returns
    -------
    float
        Horizontal symmetry index
    """
    # Reshape the input into a 2D matrix if needed
    if len(image_matrix.shape) == 1:
        image_matrix = image_matrix.reshape(8, 8)

    # Calculate the symmetry index
    rows, cols = image_matrix.shape
    symmetry_sum = 0
    for i in range(rows):
        symmetry_sum += np.sum(np.abs(image_matrix[i, :] - image_matrix[rows - 1 - i, :]))
    return symmetry_sum / (rows * cols)

def sum_of_central_region(image_matrix):
    """
    Calculate the sum of the central region in the image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input 8x8 image matrix or flattened array

    Returns
    -------
    float
        Sum of all values in the central region of the matrix
    """
    # Reshape the input array into a 2D matrix
    image_matrix_2d = image_matrix.reshape(8, 8)
    
    # Define the boundaries of the central region
    central_region = image_matrix_2d[2:6, 2:6]
    
    # Calculate the sum of values in the central region
    sum_central_region = np.sum(central_region)
    
    return sum_central_region



def column_sum_variance(image_matrix):
    """
    Calculate the variance of the sum of the columns of the image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input 8x8 image matrix

    Returns
    -------
    float
        Variance of the sum of the columns of the matrix
    """
    column_sums = np.sum(image_matrix, axis=0)
    return np.var(column_sums)


def pixel_density(image_matrix):
    """
    Calculate the pixel density of the image matrix.

    Parameters
    ----------
    image_matrix : np.ndarray
        Input 8x8 image matrix

    Returns
    -------
    float
        Pixel density (ratio of non-zero pixels to total pixels)
    """
    non_zero_pixels = np.count_nonzero(image_matrix)
    total_pixels = image_matrix.size
    return non_zero_pixels / total_pixels
    
def plot_histogram_side_by_side(feature_values_0, feature_values_1, feature_name):
    """
    Plot side-by-side histograms for two groups (Digit 0 and Digit 1) of a specific feature.

    Parameters
    ----------
    feature_values_0 : array-like
        Feature values for Digit 0.
    feature_values_1 : array-like
        Feature values for Digit 1.
    feature_name : str
        Name of the feature being visualized.

    Returns
    -------
    None
        The function displays the side-by-side histograms using Matplotlib.
    """
    
    plt.figure(figsize=(15, 5))
    plt.hist([feature_values_0, feature_values_1], bins=20, alpha=0.5,label=['Digit 0', 'Digit 1'], color=['blue', 'orange'])
    plt.xlabel(feature_name)
    plt.ylabel('Number of Occurrences')
    plt.title(f'Histogram of {feature_name} for Digits 0 and 1')
    plt.legend()
    plt.show()
    
def plot_property_pairs(feature_values_x_0, feature_values_y_0, feature_values_x_1, feature_values_y_1, feature_names):
    """
    Plots a scatter plot of two features for two groups.

    Parameters:
    - feature_values_x_0 (array-like): Feature values for group 0 on the x-axis.
    - feature_values_y_0 (array-like): Feature values for group 0 on the y-axis.
    - feature_values_x_1 (array-like): Feature values for group 1 on the x-axis.
    - feature_values_y_1 (array-like): Feature values for group 1 on the y-axis.
    - feature_names (list): List containing names of features. Should have length 2.

    Returns:
    - None

    This function plots a scatter plot where each group is represented by a different color.
    Group 0 is plotted in blue and labeled as 'Digit 0'.
    Group 1 is plotted in orange and labeled as 'Digit 1'.
    The x-axis is labeled with the name provided in feature_names[0].
    The y-axis is labeled with the name provided in feature_names[1].
    The title of the plot includes the names of the features and the groups being compared.
    """

    # Create a new figure
    plt.figure(figsize=(15, 5))
    
    # Scatter plots for group 0
    plt.scatter(feature_values_x_0, feature_values_y_0, color='blue', label='Digit 0')
    
    # Scatter plots for group 1
    plt.scatter(feature_values_x_1, feature_values_y_1, color='orange', label='Digit 1')
    
    # Set labels and title
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'Scatter Plot of {feature_names[0]} vs {feature_names[1]} for Digits 0 and 1')
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.show()

def plot_3d_scatter(feature_values_x_0, feature_values_y_0, feature_values_z_0,
                    feature_values_x_1, feature_values_y_1, feature_values_z_1,
                    feature_names):
    """
    Plot a 3D scatter plot for two groups (Digit 0 and Digit 1) using three features.

    Parameters
    ----------
    feature_values_x_0 : array-like
        Feature values for Digit 0 along the x-axis.
    feature_values_y_0 : array-like
        Feature values for Digit 0 along the y-axis.
    feature_values_z_0 : array-like
        Feature values for Digit 0 along the z-axis.
    feature_values_x_1 : array-like
        Feature values for Digit 1 along the x-axis.
    feature_values_y_1 : array-like
        Feature values for Digit 1 along the y-axis.
    feature_values_z_1 : array-like
        Feature values for Digit 1 along the z-axis.
    feature_names : list of str
        Names of the three features being visualized.

    Returns
    -------
    None
        The function displays the 3D scatter plot using Matplotlib.
    """
    # Create a new figure
    fig = plt.figure()
    
    # Set title
    fig.suptitle(f'3D Scatter Plot of {feature_names[0]} vs {feature_names[1]} vs {feature_names[2]} for Digits 0 and 1', fontsize=14)
    
    # Add 3D subplot
    ax = fig.add_subplot(projection='3d')
    
    # Scatter plot for Digit 0
    ax.scatter(feature_values_x_0, feature_values_y_0, feature_values_z_0, c='blue', label='Digit 0')
    
    # Scatter plot for Digit 1
    ax.scatter(feature_values_x_1, feature_values_y_1, feature_values_z_1, c='orange', label='Digit 1')
    
    # Set labels for axes
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    
    # Add legend
    ax.legend()
    
    # Show plot
    plt.show()


# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

#Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, linear_model, metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict



###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

# Filter indices for digits "0" and "1"
indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))

# Use the filtered indices to extract relevant data
filtered_data = data[indices_0_1]
filtered_target = digits.target[indices_0_1]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target,
                                                    test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Display incorrectly labeled digits
incorrect_predictions = np.where(predicted != y_test)[0]

# Create a subplot for all misclassified images
plt.figure(figsize=(15, 8), facecolor='lightgrey')
plt.suptitle("Test. mis-classification: expected - predicted", y=0.98, fontsize=16)

# Adjust the top value to reduce the space between the subplots and the top
plt.subplots_adjust(top=0.85)

for i, idx in enumerate(incorrect_predictions, 1):
    # Get original and incorrect labels
    original_label = y_test[idx]
    incorrect_label = predicted[idx]

    # Get the misclassified image
    misclassified_image = X_test[idx].reshape(8, 8)

    # Add subplots for each misclassified image
    plt.subplot(3,10, i)
    plt.imshow(misclassified_image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"{original_label} {incorrect_label}")
    plt.axis('off')
    
plt.show()


# Calculate the sum of values feature for each data point
sum_of_values_feature = np.apply_along_axis(sum_of_values, axis=1, arr=data)

# Calculate the pixel density feature for each data point
pixel_density_feature = np.apply_along_axis(pixel_density, axis=1, arr=data)

# Calculate the horizontal symmetry index feature for each data point
horizontal_symmetry_index_feature = np.apply_along_axis(horizontal_symmetry_index, axis=1, arr=data)

# Calculate the column sum variance feature for each data point
column_sum_variance_feature = np.apply_along_axis(column_sum_variance, axis=1, arr=data)

# Calculate the sum of the central region feature for each data point
sum_of_central_region_feature = np.apply_along_axis(sum_of_central_region, axis=1, arr=data)

# Extract feature values for both groups (digits 0 and 1)
sum_of_values_0 = sum_of_values_feature[indices_0_1][filtered_target == 0]
sum_of_values_1 = sum_of_values_feature[indices_0_1][filtered_target == 1]

pixel_density_0 = pixel_density_feature[indices_0_1][filtered_target == 0]
pixel_density_1 = pixel_density_feature[indices_0_1][filtered_target == 1]

horizontal_symmetry_index_0 = horizontal_symmetry_index_feature[indices_0_1][filtered_target == 0]
horizontal_symmetry_index_1 = horizontal_symmetry_index_feature[indices_0_1][filtered_target == 1]

column_sum_variance_0 = column_sum_variance_feature[indices_0_1][filtered_target == 0]
column_sum_variance_1 = column_sum_variance_feature[indices_0_1][filtered_target == 1]

sum_of_central_region_0 = sum_of_central_region_feature[indices_0_1][filtered_target == 0]
sum_of_central_region_1 = sum_of_central_region_feature[indices_0_1][filtered_target == 1]


# Plot histograms for each feature side by side
plot_histogram_side_by_side(sum_of_values_0, sum_of_values_1, 'Sum of Values')
plot_histogram_side_by_side(pixel_density_0, pixel_density_1, 'Pixel Density')
plot_histogram_side_by_side(horizontal_symmetry_index_0, horizontal_symmetry_index_1, 'Horizontal Symmetry Index')
plot_histogram_side_by_side(column_sum_variance_0, column_sum_variance_1, 'Column Sum Variance')
plot_histogram_side_by_side(sum_of_central_region_0, sum_of_central_region_1, 'Sum of Central Region')


# Plot pairs of properties
plot_property_pairs(sum_of_values_0, pixel_density_0, sum_of_values_1, pixel_density_1, ['Sum of Values', 'Pixel Density'])
plot_property_pairs(horizontal_symmetry_index_0, pixel_density_0, horizontal_symmetry_index_1, pixel_density_1, ['Horizontal Symmetry Index', 'Pixel Density'])
plot_property_pairs(column_sum_variance_0, sum_of_central_region_0, column_sum_variance_1, sum_of_central_region_1, ['Column Sum Variance','Sum of Central Region'])
plot_property_pairs(horizontal_symmetry_index_0, sum_of_central_region_0, horizontal_symmetry_index_1, sum_of_central_region_1, ['Horizontal Symmetry Index','Sum of Central Region'])


# Plot sets of 3D scatter plots for different feature combinations
plot_3d_scatter(pixel_density_0, horizontal_symmetry_index_0, column_sum_variance_0,
                pixel_density_1, horizontal_symmetry_index_1, column_sum_variance_1,
                ['Pixel Density', 'Horizontal Symmetry Index', 'Column Sum Variance'])
plot_3d_scatter(sum_of_central_region_0, horizontal_symmetry_index_0, sum_of_values_0,
                sum_of_central_region_1, horizontal_symmetry_index_1, sum_of_values_1,
                ['Sum of Central Region', 'Horizontal Symmetry Index', 'Sum of Values'])
plot_3d_scatter(pixel_density_0, column_sum_variance_0, sum_of_values_0,
                pixel_density_1, column_sum_variance_1, sum_of_values_1,
                ['Pixel Density', 'Column Sum Variance', 'Sum of Values'])
plot_3d_scatter(horizontal_symmetry_index_0, pixel_density_0, sum_of_values_0,
                horizontal_symmetry_index_1, pixel_density_1, sum_of_values_1,
                ['Horizontal Symmetry Index', 'Pixel Density', 'Sum of Values'])


# Choose three features for classification
featureA = sum_of_central_region_feature
featureB = pixel_density_feature
featureC = horizontal_symmetry_index_feature

# creating the X (feature) matrix
X = np.column_stack((featureA[indices_0_1], featureB[indices_0_1],featureC[indices_0_1]))
# scaling the values for better classification performance
X_scaled = preprocessing.scale(X)
# the predicted outputs
Y = digits.target[indices_0_1]
# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
logistic_classifier.fit(X_scaled, Y)
# show how good is the classifier on the training data
expected = Y 
predicted = logistic_classifier.predict(X_scaled)
print("Logistic regression using [sum_of_central_region_feature, pixel_density_feature, horizontal_symmetry_index_feature] features:\n%s\n" % (
 metrics.classification_report(
 expected,
 predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted),'\n')
# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
print("Logistic regression using [sum_of_central_region_feature, pixel_density_feature, horizontal_symmetry_index_feature] features cross validation:\n%s\n" % (
 metrics.classification_report(
 expected,
 predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))
