# section A
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 4.0])
y = np.array([-1.5, 2.0, 0.7, 5.0, 3.5, 7.5])

# Function to calculate error
def calculate_error(a, b):
    return np.sum((a * x + b - y) ** 2)

# Gradient descent
def gradient_descent(x, y, learning_rate, max_iterations):
    a = 1.0
    b = 1.0
    error_history = []
    parameter_values = []

    for i in range(max_iterations):
        # Calculate gradients
        gradient_a = 2 * np.sum((a * x + b - y) * x)
        gradient_b = 2 * np.sum(a * x + b - y)

        # Update parameters
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b

        # Calculate error
        error = calculate_error(a, b)
        error_history.append(error)
        parameter_values.append((a, b))  # Save parameter values

    return a, b, error_history, parameter_values

# Hyperparameters
learning_rate = 0.001
max_iterations = 1000

# Perform gradient descent
a_final, b_final, error_history, parameter_values = gradient_descent(x, y, learning_rate, max_iterations)

# Print final parameter values
print("a =", a_final,) 
print("b =", b_final,"\n")


# Plot data and final model
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data')
plt.plot(x, a_final * x + b_final, color='red', label='Final Model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Final Model')
plt.legend()
plt.axis('equal')
plt.show()

# Plot error as a function of iteration number
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iterations + 1), error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error as a Function of Iteration Number')
plt.grid(True)
plt.show()

# Function to calculate error surface
def calculate_error_surface(x, y, a_range, b_range):
    errors = np.zeros((len(a_range), len(b_range)))
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            errors[i, j] = calculate_error(a, b)
    return errors

# Hyperparameters for error surface plot
a_range_surface = np.linspace(-2, 4, 100)
b_range_surface = np.linspace(-2, 4, 100)
A_surface, B_surface = np.meshgrid(a_range_surface, b_range_surface)
errors_surface = calculate_error_surface(x, y, a_range_surface, b_range_surface)

# Adjust view angles for better visualization
view_angles = [(30, 30), (60, 60), (90, 90), (120, 120)]

# Create subplots for all views
fig = plt.figure(figsize=(20, 15))

# Plot error surface with iteration path for each view angle
for i, angle in enumerate(view_angles):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    ax.plot_surface(A_surface, B_surface, errors_surface, cmap='viridis',alpha=0.5)

    # Plot iteration path
    for j in range(len(parameter_values) - 1):
        ax.plot([parameter_values[j][0], parameter_values[j + 1][0]],
                [parameter_values[j][1], parameter_values[j + 1][1]],
                [error_history[j], error_history[j + 1]], color='red')

    ax.scatter(a_final, b_final, calculate_error(a_final, b_final), color='red', s=100)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Error')
    ax.set_title(f'Error Surface (View: {angle})')
    ax.view_init(angle[0], angle[1])

plt.show()


# section C
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sin, diff
from scipy.interpolate import interp1d


def gradient_descent_sin(F, E, DA, DB, x_, y_, a0, b0, learning_rate, num_iterations):
    
    error_list = []
    a_list = []
    b_list = []
    
    for i in range(num_iterations):
        error = np.sum(E(a0, b0, x_, y_))
        error_list.append(error)
        
        da_val = np.sum(DA(a0, b0, x_, y_))
        db_val = np.sum(DB(a0, b0, x_, y_))
        a0 -= learning_rate * da_val
        b0 -= learning_rate * db_val
        
        # Save the parameter values in each iteration
        a_list.append(a0)
        b_list.append(b0)
    
    return a0, b0, a_list, b_list, error_list


def plot_data_and_model_sin(x, y, a, b, F):
    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Final Model')

    y_sin = F(a, b, x)
    x_new = np.linspace(x.min(), x.max(), num=1000)
    f_interp = interp1d(x, y_sin, kind='cubic')
    y_sin_interp = f_interp(x_new)
    plt.plot(x_new, y_sin_interp, c='r', label="Final Model")
    plt.legend()
    plt.show()


def plot_errors(error_list):
    plt.figure()
    plt.plot(range(len(error_list)), error_list)
    plt.title('Error as a Function of Iteration Number')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()



def calculate_error_surface(x, y, a_range, b_range):
    errors = np.zeros((len(a_range), len(b_range)))
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            errors[i, j] = np.sum((a * np.sin(b * x) - y) ** 2)
    return errors



x_ = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1.,
         1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
y_ = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917, 1.14600963,
         0.15938811, -3.09848048, -3.67902427, -1.84892687, -0.11705947,
         3.14778203, 4.26365256, 2.49120585, 0.55300516, -2.105836 , -2.68898773,
         -2.39982575, -0.50261972, 1.40235643, 2.15371399])



# Define symbolic variables
x, y, a, b = symbols('x, y, a, b')



# Define the model function
f = a * sin(b * x)
F = lambdify([a, b, x], f, 'numpy')



# Define the error for a single data point
e = ((a * sin(b * x)) - y) ** 2



# Create a function that can operate on numpy arrays
E = lambdify([a, b, x, y], e, 'numpy')



# Define the gradient components by symbolically deriving the error
da = diff(e, a)
db = diff(e, b)

DA = lambdify([a, b ,x, y], da, 'numpy')
DB = lambdify([a, b, x, y], db, 'numpy')



a0, b0, a_list, b_list, error_list = gradient_descent_sin(F, E, DA, DB, x_, y_, 1, 1, 0.001, 1000)

print("a =", a0)
print("b =", b0, "\n")
# Plot data and final model
plot_data_and_model_sin(x_, y_, a0, b0, F)



# Plot error as a function of iteration number
plot_errors(error_list)


# Hyperparameters for error surface plot
a_range_surface = np.linspace(-2, 4, 100)
b_range_surface = np.linspace(-2, 4, 100)
A_surface, B_surface = np.meshgrid(a_range_surface, b_range_surface)
errors_surface = calculate_error_surface(x_, y_, a_range_surface, b_range_surface)

# Create subplots for all views
fig = plt.figure(figsize=(20, 15))

# Plot error surface with iteration path for each view angle
for i, angle in enumerate([(30, 30), (60, 60), (90, 90), (120, 120)]):
    ax = fig.add_subplot(1, 5, i+1, projection='3d')
    ax.plot_surface(A_surface, B_surface, errors_surface, cmap='viridis',alpha=0.5)
    ax.scatter(a_list, b_list, error_list, color='red', s=50)
    ax.plot(a_list, b_list, error_list, color='red')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Error')
    ax.set_title(f'Error Surface (View: {angle})')
    ax.view_init(angle[0], angle[1])

plt.show()



# section D
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define the model function
def sin_model(x, a, b):
    return a * np.sin(b * x)


# Generate example data
x_data = np.array([-5., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
y_data = np.array([-2.16498306, -1.53726731, 1.67075645, 2.47647932, 4.49579917, 1.14600963, 0.15938811, -3.09848048, -3.67902427, -1.84892687, -0.11705947, 3.14778203, 4.26365256, 2.49120585, 0.55300516, -2.105836, -2.68898773, -2.39982575, -0.50261972, 1.40235643, 2.15371399])


# Fit the model to the data
popt, pcov = curve_fit(sin_model, x_data, y_data, p0=[1, 1])


# Extract the optimized parameters
a_opt, b_opt = popt


# Plot the model with the optimized parameters overlaid on the data
plt.figure()
plt.scatter(x_data, y_data, label="Data")
plt.plot(x_data, sin_model(x_data, a_opt, b_opt), c='r', label="Model with Optimized Parameters")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Model with Optimized Parameters')
plt.legend()
plt.show()


# Print the optimized parameters
print("a =", a_opt)
print("b =", b_opt, "\n")

# section E
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, sin, diff
from scipy.optimize import curve_fit
from time import time

def generate_noisy_data(x, a, b, c, noise_level=0.1):
    y = a * np.sin(b * x + c)
    noise =  [noise_level for _ in range(len(x))]
    return y + noise

def calculate_error_surface(x, y, a_range, b_range):
    errors = np.zeros((len(a_range), len(b_range)))
    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            errors[i, j] = np.sum((y - a * np.sin(b * x))**2)
    return errors

def plot_errors(error_list):
    plt.figure()
    plt.plot(range(len(error_list)), error_list)
    plt.title('Error Value')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()
    
def plot_error_surface_nonlinear(a_list, b_list, c_list, error_list, x_, y_, E):
    fig = plt.figure(figsize=(18, 6))
    titles = ['a and b', 'a and c', 'b and c']
    parameters = [(a_list, b_list), (a_list, c_list), (b_list, c_list)]
    
    for i, (param1, param2) in enumerate(parameters, start=1):
        a_range = np.linspace(param1[0]-0.5, param1[0]+0.5, 100)
        b_range = np.linspace(param2[0]-0.5, param2[0]+0.5, 100)
        A, B = np.meshgrid(a_range, b_range)

        Z = np.zeros_like(A)
        for j in range(A.shape[0]):
            for k in range(A.shape[1]):
                Z[j, k] = np.sum(E(A[j, k], B[j, k], c_list[0], x_, y_))

        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.plot_surface(A, B, Z, cmap='viridis', alpha=0.5)
        ax.scatter(param1, param2, error_list, c='red')
        ax.set_title(f"Error Surface for {titles[i-1]}")
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('Error')

    plt.tight_layout()
    plt.show()




def gradient_descent_nonlinear(F, E, DA, DB, DC, x_, y_, a0, b0, c0,
                               learning_rate, num_iterations):
    error_list = []
    a_list = []
    b_list = []
    c_list = []
    
    for i in range(num_iterations):
        error = np.sum(E(a0, b0, c0, x_, y_))
        error_list.append(error)
        
        da_val = np.sum(DA(a0, b0, c0, x_, y_))
        db_val = np.sum(DB(a0, b0, c0, x_, y_))
        dc_val = np.sum(DC(a0, b0, c0, x_, y_))
        a0 -= learning_rate * da_val
        b0 -= learning_rate * db_val
        c0 -= learning_rate * dc_val
        
        a_list.append(a0)
        b_list.append(b0)
        c_list.append(c0)
    
    return a0, b0, c0, a_list, b_list, c_list, error_list

def plot_data_and_nonlinear_model(x, y, a, b, c, F):
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Model')
    plt.scatter(x, y, color='blue', label='data')
    plt.plot(x, F(a, b, c, x), color='red', label='model')
    plt.legend()
    plt.show()

def nonlinear_model(x, a, b, c):
    return a * np.sin(b * x + c)

x_ = np.linspace(-5, 5, 20)
y_ = generate_noisy_data(x_, 0.5, 0.6, 3)

x_ = np.array(x_)
y_ = np.array(y_)

# Define symbolic variables
x, y, a, b, c = symbols('x, y, a, b, c')

# Define the model function
f = a * sin(b * x + c)
F = lambdify([a, b, c, x], f, 'numpy')

# Define the error for a single data point
e = ((a * sin(b * x + c)) - y)**2

# Create a function that can operate on numpy arrays
E = lambdify([a, b, c, x, y], e, 'numpy')

# Define the gradient components by symbolically deriving the error
da = diff(e, a)
db = diff(e, b)
dc = diff(e, c)

DA = lambdify([a, b, c, x, y], da, 'numpy')
DB = lambdify([a, b, c, x, y], db, 'numpy')
DC = lambdify([a, b, c, x, y], dc, 'numpy')

t0 = time()
a0, b0, c0, a_list, b_list, c_list, error_list = gradient_descent_nonlinear(F, E, DA, DB, DC, x_, y_, 0.3, 0.4, 0.5, 0.001, 1000)
print("Gradient Descent time:", (time()-t0))
print("gradient descent parameters:\na = {}\nb = {}\nc = {}\n".format(a0, b0, c0))

plot_data_and_nonlinear_model(x_, y_, a0, b0, c0, F)
plot_errors(error_list)
plot_error_surface_nonlinear(a_list, b_list, error_list, c0, x_, y_, E)
t0 = time()
popt, pcov = curve_fit(nonlinear_model, x_, y_)
print("Curve Fit time:", (time()-t0))
print("curve_fit parameters:\na = {}\nb = {}\nc = {}\n".format(popt[0], popt[1], popt[2]))

# Plot data before and after adding noise
plt.figure(figsize=(10, 6))
plt.scatter(x_, y_, color='blue', label='Noisy Data')
plt.plot(x_, 0.5 * np.sin(0.6 * x_ + 3), color='green', linestyle='--', label='Original Model')
plt.legend()
plt.title('Data Before and After Adding Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot models on noisy data
plt.figure(figsize=(10, 6))
plt.scatter(x_, y_, color='blue', label='Noisy Data')
plt.plot(x_, nonlinear_model(x_, a0, b0, c0), color='red', label='GD Model')
plt.plot(x_, nonlinear_model(x_, *popt), color='green', label='Curve Fit Model')
plt.legend()
plt.title('Models on Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()