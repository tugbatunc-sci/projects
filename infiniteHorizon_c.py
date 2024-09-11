import numpy as np
import matplotlib.pyplot as plt

# Define the exact solution for the value function
def value_function(x1, x2):
    return np.sqrt(3)/2 * x1**2 + np.sqrt(3)/2 * x2**2 + x1 * x2

# Define the exact optimal control
def optimal_control(x1, x2):
    return -np.sqrt(3) * x2 - x1

# Generate a grid for visualization
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)

# Compute the value function and optimal control over the grid
V = value_function(X1, X2)
D = optimal_control(X1, X2)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Value Function
c = ax[0].contourf(X1, X2, V, levels=50, cmap='viridis')
fig.colorbar(c, ax=ax[0])
ax[0].set_title('Value Function')
ax[0].set_xlabel('$x_1$')
ax[0].set_ylabel('$x_2$')

# Optimal Control
c = ax[1].contourf(X1, X2, D, levels=50, cmap='inferno')
fig.colorbar(c, ax=ax[1])
ax[1].set_title('Optimal Control')
ax[1].set_xlabel('$x_1$')
ax[1].set_ylabel('$x_2$')

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Simulate X-TFC approximations (for demonstration)
def simulate_xtfc_solution(x1, x2):
    # Simulated approximations for value function and optimal control
    V_xtfc = value_function(x1, x2) * np.random.uniform(0.98, 1.02, size=x1.shape)
    D_xtfc = optimal_control(x1, x2) * np.random.uniform(0.98, 1.02, size=x1.shape)
    return V_xtfc, D_xtfc

# Compute the X-TFC approximated solutions
V_xtfc, D_xtfc = simulate_xtfc_solution(X1, X2)

# Compute absolute errors
V_error = np.abs(V - V_xtfc)
D_error = np.abs(D - D_xtfc)

# Plotting in 3D
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(X1, X2, V, cmap='viridis', edgecolor='none')
ax.set_title('Exact Value Function')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('V')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(X1, X2, V_xtfc, cmap='viridis', edgecolor='none')
ax.set_title('X-TFC Approximation (Value Function)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('V')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(X1, X2, D, cmap='inferno', edgecolor='none')
ax.set_title('Exact Optimal Control')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('D')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(X1, X2, D_xtfc, cmap='inferno', edgecolor='none')
ax.set_title('X-TFC Approximation (Optimal Control)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('D')

plt.tight_layout()
plt.show()

# Plotting Absolute Errors in 3D
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X1, X2, V_error, cmap='plasma', edgecolor='none')
ax.set_title('Absolute Error (Value Function)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Error')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X1, X2, D_error, cmap='plasma', edgecolor='none')
ax.set_title('Absolute Error (Optimal Control)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Error')

plt.tight_layout()
plt.show()