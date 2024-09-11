import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# We will define our function given in the question.

def f(x, y):
    return 1.8 - np.exp(-0.1 * (2.5 * (x + 3) ** 2 + (y + 3) ** 2)) - 1.5 * np.exp(-0.05 * (2.5 * (x - 3) ** 2 + (y - 3) ** 2))

# Creating a 2D grid with coordinates -8 to 8 for each dimension with a step size of 0.1
x = np.arange(-8, 8, 0.1)
y = np.arange(-8, 8, 0.1)
X, Y = np.meshgrid(x, y)

# Evaluating the function on the grid
Z = f(X, Y)

# Implementing stochastic gradient descent
# We are implementing SGD from directly numpy.gradient
# Not as the way we apply in the class explicitly.

def stochastic_gradient_descent(f, lr=0.1, max_iter=1000, batch_size=10):
    theta_values = []  # Creating an empty list
    for point in initial_points:
        theta = np.array(point, dtype=float)  # All theta points are set to be float.
        for _ in range(max_iter):
            indices = np.random.choice(len(x) * len(y), size=batch_size, replace=False)
            x_batch = X.flatten()[indices]
            y_batch = Y.flatten()[indices]
            
            # Evaluating gradients using np.gradient
            gradients = np.gradient(f(x_batch, y_batch), 0.1)
            grad = np.column_stack((gradients[0], gradients[1])) 
            
            theta -= lr * np.mean(grad, axis=0).astype(float)  # To match the data type, we insert type again.
            theta_values.append(theta.copy())  # Appending our list with theta values.
            
            # Updating function values in Z at closest grid points to the updated coordinates
            updated_x, updated_y = theta[0], theta[1]
            closest_x_idx = np.argmin(np.abs(x - updated_x))
            closest_y_idx = np.argmin(np.abs(y - updated_y))
            Z[closest_y_idx, closest_x_idx] = f(updated_x, updated_y)  # Update Z with new function value
    return np.array(theta_values)

# We will find the gradient at a specific point using RegularGridInterpolator as expected.
interpolator = RegularGridInterpolator((x, y), Z.T)

# Applying our defined stochastic gradient descent function to obtain our list.
updated_values = stochastic_gradient_descent(f)

# We will define 3 initial points for stochastic gradient descent
initial_points = [(-6, -6), (0, 0), (5, 5)]

# Printing interpolated values at initial points
# Interpolated values are the estimated value of the function at that point.
# Updated values are results of Stochastic Gradient Descent where iteratively updates the (x, y) randomly.

print("Interpolated values:")
for point in initial_points:
    interpolated_value = interpolator(point) 
    print(f"Interpolated value at {point}: {interpolated_value}")

# Printing updated values at initial points
print("\nUpdated values:")
for i, point in enumerate(initial_points):
    print(f"Updated value at {point}: {updated_values[(i+1)*1000 - 1]}")

# We will make a contour plot to show the updated values
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of f(x, y) with updated values')
plt.scatter(updated_values[:, 0], updated_values[:, 1], c='red', marker='x', label='Updated Values')
for point in initial_points:
    plt.scatter(point[0], point[1], c='blue', marker='o', label='Initial Points')
plt.legend()
plt.show()

# Variation of results with respect to different learning rates
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    # Applying our defined stochastic gradient descent function to obtain our list.
    updated_values = stochastic_gradient_descent(f, lr=lr)
    
    # Printing interpolated values at initial points
    print(f"Interpolated values for Learning Rate: {lr}")
    for point in initial_points:
        interpolated_value = interpolator(point)
        print(f"Interpolated value at {point}: {interpolated_value}")
    
    # Printing updated values at initial points
    print(f"\nUpdated values for Learning Rate: {lr}")
    for i, point in enumerate(initial_points):
        print(f"Updated value at {point}: {updated_values[(i+1)*1000 - 1]}")

# We will make a contour plot to show the updated values wrt varying lr's
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of f(x, y) with updated values')
plt.scatter(updated_values[:, 0], updated_values[:, 1], c='red', marker='x', label='Updated Values')
for point in initial_points:
    plt.scatter(point[0], point[1], c='blue', marker='o', label='Initial Points')
plt.legend()
plt.show()

for lr in learning_rates:
    # Apply stochastic gradient descent and store updated values
    updated_values = stochastic_gradient_descent(f, lr=lr)
    
    # Calculate loss using the function value difference
    loss_values = []
    for i, point in enumerate(initial_points):
        interpolated_value = interpolator(point)
        updated_value = f(updated_values[(i+1)*1000 - 1, 0], updated_values[(i+1)*1000 - 1, 1])
        loss = np.abs(interpolated_value - updated_value)
        loss_values.append(loss)
    
    # Print loss for each initial point at the current learning rate
    print(f"Loss for Learning Rate: {lr}")
    for i, point in enumerate(initial_points):
        print(f"Loss at {point}: {loss_values[i]}")

# Defining the range for learning rates
min_lr = 0.001
max_lr = 10.0
num_lr = 15  # Number of learning rates

learning_rates = np.linspace(min_lr, max_lr, num_lr)

for lr in learning_rates:
    updated_values = stochastic_gradient_descent(f, lr=lr)
    
    # Calculating loss using the function value difference
    loss_values = []
    for i, point in enumerate(initial_points):
        interpolated_value = interpolator(point)
        updated_value = f(updated_values[(i+1)*1000 - 1, 0], updated_values[(i+1)*1000 - 1, 1])
        loss = np.abs(interpolated_value - updated_value)
        loss_values.append(loss)
    
    # Print loss for each initial point at the current learning rate
    print(f"Loss for Learning Rate: {lr}")
    for i, point in enumerate(initial_points):
        print(f"Loss at {point}: {loss_values[i]}")

