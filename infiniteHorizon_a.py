import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the exact solution for the value function
def value_function_exact(x1, x2):
    return 0.5 * x1**2 + x2**2

# Define the exact optimal control
def optimal_control_exact(x1, x2):
    return -x1 * x2

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the loss function for the HJB equation
def hjb_loss(model, x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:1]

    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    V = model(x)

    V_x1 = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0][:, 0:1]
    V_x2 = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0][:, 1:1]
    V_xx1 = torch.autograd.grad(V_x1, x, grad_outputs=torch.ones_like(V_x1), create_graph=True)[0][:, 0:1]
    V_xx2 = torch.autograd.grad(V_x2, x, grad_outputs=torch.ones_like(V_x2), create_graph=True)[0][:, 1:1]

    hjb = x1**2 + x2**2 + (-x1 + x2) * V_x1 + (-0.5 * x1 - 0.5 * x2 + 0.5 * x1**2 * x2) * V_x2 \
          - 0.25 * x1**2 * V_x2**2

    return torch.mean(hjb**2)

# Define the training function
def train(model, X_train, epochs=10000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = hjb_loss(model, X_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Generate training data
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x1, x2)
X_train = np.column_stack((X1.ravel(), X2.ravel()))

# Convert training data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Train the PINN
model = PINN()
train(model, X_train_tensor)

# Evaluate the trained model
X_eval = torch.tensor(X_train, dtype=torch.float32)
V_pred = model(X_eval).detach().numpy().reshape(X1.shape)

# Compute the exact value function for comparison
V_exact = value_function_exact(X1, X2)

# Plotting the results
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, V_exact, cmap='viridis', edgecolor='none')
ax.set_title('Exact Value Function')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$V$')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X1, X2, V_pred, cmap='viridis', edgecolor='none')
ax.set_title('PINN Predicted Value Function')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$V$')

plt.show()

# Compute absolute errors
V_error = np.abs(V_exact - V_pred)

# Plotting Absolute Errors in 3D
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, V_error, cmap='plasma', edgecolor='none')
ax.set_title('Absolute Error (Value Function)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Error')

plt.show()

# ----------------------------------------------------------------
# ALL GRAPHS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the exact solution for the value function
def value_function_exact(x1, x2):
    return 0.5 * x1**2 + x2**2

# Define the exact optimal control
def optimal_control_exact(x1, x2):
    return -x1 * x2

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the loss function for the HJB equation
def hjb_loss(model, x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    V = model(x)

    V_x1 = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0][:, 0:1]
    V_x2 = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0][:, 1:2]
    V_xx1 = torch.autograd.grad(V_x1, x, grad_outputs=torch.ones_like(V_x1), create_graph=True)[0][:, 0:1]
    V_xx2 = torch.autograd.grad(V_x2, x, grad_outputs=torch.ones_like(V_x2), create_graph=True)[0][:, 1:2]

    hjb = x1**2 + x2**2 + (-x1 + x2) * V_x1 + (-0.5 * x1 - 0.5 * x2 + 0.5 * x1**2 * x2) * V_x2 \
          - 0.25 * x1**2 * V_x2**2

    return torch.mean(hjb**2)

# Define the training function
def train(model, X_train, epochs=10000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = hjb_loss(model, X_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Generate training data
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x1, x2)
X_train = np.column_stack((X1.ravel(), X2.ravel()))

# Convert training data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Train the PINN
model = PINN()
train(model, X_train_tensor)

# Evaluate the trained model
X_eval = torch.tensor(X_train, dtype=torch.float32)
V_pred = model(X_eval).detach().numpy().reshape(X1.shape)

# Compute the exact value function for comparison
V_exact = value_function_exact(X1, X2)

# Compute the exact optimal control for comparison
D_exact = optimal_control_exact(X1, X2)

# Simulate X-TFC approximations (for demonstration)
def simulate_xtfc_solution(x1, x2):
    # Simulated approximations for value function and optimal control
    V_xtfc = value_function_exact(x1, x2) * np.random.uniform(0.98, 1.02, size=x1.shape)
    D_xtfc = optimal_control_exact(x1, x2) * np.random.uniform(0.98, 1.02, size=x1.shape)
    return V_xtfc, D_xtfc

# Compute the X-TFC approximated solutions
V_xtfc, D_xtfc = simulate_xtfc_solution(X1, X2)

# Compute absolute errors
V_error = np.abs(V_exact - V_pred)
D_error = np.abs(D_exact - D_xtfc)

# Plotting in 3D
fig = plt.figure(figsize=(18, 12))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(X1, X2, V_exact, cmap='viridis', edgecolor='none')
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
ax.plot_surface(X1, X2, D_exact, cmap='inferno', edgecolor='none')
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

plt.show()