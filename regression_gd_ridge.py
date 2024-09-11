import numpy as np
import matplotlib.pyplot as plt
from os import sys

np.random.seed(123)


class RidgeRegression:

	def __init__(self, theta_init, lamb=1, lr=0.01, max_iter=100, tol=1e-04):
		self.lamb = lamb
		self.lr = lr
		self.max_iter = max_iter
		self.tol = tol
		self.theta = theta_init


	# Squared error loss
	def calc_loss(self, X, y):
		N = len(y)
		yhat = X@self.theta # ypred
		w = self.theta[1]
		loss = (np.sum((yhat - y)**2) + self.lamb*w.T@w) / (2*N)
		return loss
	
	def calc_grad(self, X, y):
		N = len(y)
		yhat = X@self.theta		
		gradient = X.T@(yhat - y) / N
		return gradient	
			

	def fit(self, X, y):
	
		loss_history = np.zeros(self.max_iter)
		
		for i in range(self.max_iter):		
			regularization = self.lamb*self.theta[1] / len(y)
			grad = self.calc_grad(X, y) + regularization 		
			step = -self.lr*grad
			self.theta += step
			loss_history[i] = self.calc_loss(X, y)
			if i%10 == 0:
				print("Current Iteration: {}, \
					   Current Loss: {}".format(i, loss_history[i]))
			if np.linalg.norm(grad) < self.tol:
				print("Converged at iteration {}".format(i))
				break
					
		return self.theta, loss_history					


	def predict(self, X):
		y_pred = X@self.theta
		return y_pred		



# True model parameters
m_true = -0.8567
b_true = 2.896

# Generate data.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1 + 0.5*np.random.rand(N)
y = m_true*x + b_true
y += yerr*np.random.randn(N)



# Convert to shape (N, 1)
x = x[:, np.newaxis]
y_train = y[:, np.newaxis]

# Convert to shape (N, 2) with first column filled with ones
X_train = np.c_[np.ones(len(x)), x]


theta_init = np.random.randn(2, 1)
#print("Initial values:", theta_init)


# Create model
max_iter = 1000
model = RidgeRegression(theta_init, lr=0.01, max_iter=max_iter)

# Fit model
theta, cost = model.fit(X_train, y_train)


Xt = 2*np.random.rand(2, 1)
X_test = np.c_[np.ones((len(Xt))), Xt]
print("Test point:", X_test)

prediction = model.predict(X_test)

print("prediction:", prediction)
print("theta:", theta)


fig = plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt=".b", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true*x0 + b_true, "k", alpha=0.3, lw=3)
plt.plot(Xt, prediction, ".r")
plt.plot(x0, theta[1]*x0 + theta[0], "r", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")


iterations = np.arange(1, max_iter+1)

fig = plt.figure()
plt.plot(iterations, cost)
plt.show()