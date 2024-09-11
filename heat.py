import tensorflow as tf
import numpy as np

# Define the PINNs model
class HeatPINN(tf.keras.Model):
    def __init__(self):
        super(HeatPINN, self).__init__()
        # Define the neural network architecture
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, t):
        # Concatenate x and t as input
        input_tensor = tf.concat([x, t], 1)
        # Pass through neural network
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Define the PINNs loss function
# Define the PINNs loss function
def PINN_loss(model, x, t, k, f, T1, T2):
    with tf.GradientTape(persistent=True) as tape:
        # Watch the input variables
        tape.watch(x)
        tape.watch(t)
        # Compute predictions and gradients
        u = model(x, t)
        du_dx = tape.gradient(u, x)
        du_dt = tape.gradient(u, t)
        d2u_dx2 = tape.gradient(du_dx, x)

    # Compute PDE residual
    pde_residual = du_dt - k * d2u_dx2

    # Compute initial condition residual
    initial_residual = model(x, tf.zeros_like(t)) - f(x)

    # Compute boundary condition residuals
    boundary_residual1 = model(tf.zeros_like(x), t) - T1
    boundary_residual2 = model(L * tf.ones_like(x), t) - T2

    # Compute total loss
    total_loss = tf.reduce_mean(tf.square(pde_residual)) + \
                 tf.reduce_mean(tf.square(initial_residual)) + \
                 tf.reduce_mean(tf.square(boundary_residual1)) + \
                 tf.reduce_mean(tf.square(boundary_residual2))

    return total_loss

# Convert training data to TensorFlow tensors
x_train_tf = tf.constant(x_train, dtype=tf.float32)
t_train_tf = tf.constant(t_train, dtype=tf.float32)

# Training loop
for epoch in range(10000):
    with tf.GradientTape() as tape:
        loss = PINN_loss(model, x_train_tf, t_train_tf, k, f, T1, T2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch % 100 == 0:
        print("Epoch {}: Total Loss = {}".format(epoch, loss.numpy()))

# Evaluate PINNs solution
u_pinns = model(x_train_tf, t_train_tf).numpy()

# Define analytical solution
def u_analytical(x):
    return ((T2 - T1) / L) * x + T1

# Evaluate analytical solution
u_analytical_values = u_analytical(x_train)

# Compute mean squared error
mse = np.mean((u_pinns - u_analytical_values)**2)

print("Mean Squared Error between PINNs and Analytical Solution:", mse)

import matplotlib.pyplot as plt

# Plot PINNs solution and analytical solution
plt.figure(figsize=(10, 6))
plt.plot(x_train, u_pinns, label='PINNs Solution', linestyle='--')
plt.plot(x_train, u_analytical_values, label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of PINNs Solution and Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()
