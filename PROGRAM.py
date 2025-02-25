
import GPy
import numpy as np
import matplotlib.pyplot as plt


#np.random.seed(42)

data = np.loadtxt('datafile.dat')


X = data[:,0].reshape(-1,1)
Y = data[:,1].reshape(-1,1)
Y_imag = data[:,2].reshape(-1,1)



#kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
kernel = GPy.kern.RBF(input_dim=1, variance=np.random.uniform(0.1, 10), lengthscale=np.random.uniform(0.1, 10))

kernel_imag = GPy.kern.RBF(input_dim=1, variance=np.random.uniform(0.1, 10), lengthscale=np.random.uniform(0.1, 10))
#kernel_imag = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
#kernel = GPy.kern.Matern32(1)


#kernel_imag.variance.set_prior(GPy.priors.Gamma(2., 0.5))  # Example, set prior explicitly
#kernel_imag.lengthscale.set_prior(GPy.priors.Gamma(2., 0.5))
#kernel.variance.set_prior(GPy.priors.Gamma(2., 0.5))  # Example, set prior explicitly
#kernel.lengthscale.set_prior(GPy.priors.Gamma(2., 0.5))
##print("Initial kernel parameters:", kernel)

X_mean_old = np.mean(X)
Y_mean_old = np.mean(Y)
Y_imag_mean_old = np.mean(Y_imag)
X_std_old = np.std(X)
Y_std_old = np.std(Y)
Y_imag_std_old = np.std(Y_imag)
print("Initial kernel parameters:", X,Y,Y_imag,Y_std_old, Y_imag_std_old)


X_old=X
Y_old=Y
Y_imag_old=Y_imag



X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)
Y_imag = (Y_imag - np.mean(Y_imag)) / np.std(Y_imag)

#X_mean_new = np.mean(X)
#Y_mean_new = np.mean(Y)
#X_std_new = np.std(X)
#Y_std_new = np.std(Y)

#Z = np.linspace(min(X), max(X), 3).reshape(-1, 1)
m = GPy.models.GPRegression(X,Y,kernel)
m.Gaussian_noise.variance.fix(0.000000)

#m = GPy.models.SparseGPRegression(X, Y, kernel, Z=Z)
#m.Gaussian_noise.constrain_positive()
#m.Mat32.constrain_positive()


#m.likelihood.variance = 1e-1
from IPython.display import display
#print(m[''])

#fig = m.plot()
#plt.show()





#m.optimize(messages=True)
m.optimize_restarts(messages=False, max_iters=1000, optimizer='bfgs', num_restarts=40)
display(m)
fig = m.plot()
plt.show()
#



m_imag = GPy.models.GPRegression(X,Y_imag,kernel_imag)
m_imag.Gaussian_noise.variance.fix(0.000000)
m_imag.optimize_restarts(messages=False, max_iters=1000, optimizer='bfgs', num_restarts=40)
#print (m.gradient)

#m.optimize_restarts(num_restarts = 15)

display(m_imag)
#
#
fig = m_imag.plot()
plt.show()
#

####EXTRACING THE VALUE OF ENERGY FOR A GIVEN VALUE OF ETA

# Define new input point x*
#X_new = np.linspace(1.2e-3, 1.85e-3, 100)   # Example new input
# Normalize new input
#X_new_norm = (X_new - X_mean_old) / X_std_old
#X_new_norm = X_new_norm.reshape(-1, 1)
# Predict normalized output
#Y_new_norm, Y_new_var_norm = m.predict(X_new_norm)

# Denormalize output
#Y_new = Y_new_norm * Y_std_old + Y_mean_old

#for x in np.linspace(1.2e-3, 1.85e-3, 100):
#  print(x)
#print("Predicted y:", Y_new)

#plt.plot(X_new, Y_new)
#plt.scatter(X_old, Y_old,marker='v')
#plt.show()

#### CALCULATING THE DERIVATIVE OF THE SUUROGATE MODEL

K = kernel.K(X,X)
K_imag = kernel_imag.K(X,X)


from scipy.linalg import cho_solve, cholesky

# Compute Cholesky decomposition
L = cholesky(K + np.eye(len(X)) * 1e-6, lower=True)  # Add jitter for stability
L_imag = cholesky(K_imag + np.eye(len(X)) * 1e-6, lower=True)

# Solve for alpha (K * alpha = Y)
alpha = cho_solve((L, True), Y)
alpha_imag = cho_solve((L_imag, True), Y_imag)

def gp_surrogate_model(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)

    # Extract lengthscale parameter
    # Compute gradient
    y_surrogate = k_star @ alpha

    # Convert back to original scale (denormalization)
    yy = y_surrogate * Y_std_old + Y_mean_old

    return yy.flatten()


def gp_surrogate_covariance(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)
    k_star_star = kernel.K(x_star_norm, x_star_norm)

    L = cholesky(K + np.eye(len(X)) * 1e-10, lower=True)
    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star.T)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star - k_star @ v
    # Extract lengthscale parameter
    # Compute gradient

    # Convert back to original scale (denormalization)
    yy = y_surrogate_cov * Y_std_old**2
    return yy.flatten()








def gp_surrogate_model_imag(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star_imag = kernel_imag.K(x_star_norm, X)

    # Extract lengthscale parameter
    # Compute gradient
    y_surrogate = k_star_imag @ alpha_imag

    # Convert back to original scale (denormalization)
    yy_imag = y_surrogate * Y_imag_std_old + Y_imag_mean_old

    
    return yy_imag.flatten()

def gp_surrogate_covariance_imag(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star_imag = kernel_imag.K(x_star_norm, X)
    k_star_star_imag = kernel_imag.K(x_star_norm, x_star_norm)

    L = cholesky(K_imag + np.eye(len(X)) * 1e-10, lower=True)
    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star_imag.T)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star_imag - k_star_imag @ v
 

    # Compute gradient

    # Convert back to original scale (denormalization)
    yy_imag = y_surrogate_cov * Y_imag_std_old**2

    return yy_imag.flatten()





x_min = np.min(X_old)
x_max = np.max(X_old)

# Create an array of 100 evenly spaced points between x_min and x_max
new_points = np.linspace(x_min, x_max, 500).reshape(-1, 1)
#new_points = X_old

#X_test = np.linspace(1.2e-3, 1.85e-3, 100).reshape(-1,1)
#print(X_test)
Y_list = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list)
Y_list_imag = np.array([gp_surrogate_model_imag(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list_imag)



Y_list_cov = np.array([gp_surrogate_covariance(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list)
Y_list_imag_cov = np.array([gp_surrogate_covariance_imag(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list_imag)




std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev



std_dev = np.sqrt(Y_list_imag_cov)
upper_bound_imag = Y_list_imag + 1.96 * std_dev
lower_bound_imag = Y_list_imag - 1.96 * std_dev
# Ensure Y_list is in correct shape
#Y_list = Y_list.flatten()  # Convert to 1D array if needed
#Y_list_cov = Y_list_cov.flatten()
# Plot the surrogate model predictions
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
#plt.scatter(X_old, Y_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()
#


#plt.figure(figsize=(8, 5))
#plt.plot(new_points.flatten(), std_dev.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), std_dev.flatten(), label="GP Surrogate Model", color='b')
##plt.scatter(X_old, Y_old, color='r', marker='x', label="Training Data")  # Original data points
#plt.xlabel("X_test (New Inputs)")
#plt.ylabel("Predicted Y")
#plt.title("Gaussian Process Surrogate Model")
#plt.legend()
#plt.grid()
#plt.show()
#

plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list_imag.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list_imag.flatten(), label="GP Surrogate Model", color='b')
plt.fill_between(new_points.flatten(), lower_bound_imag, upper_bound_imag, color='b', alpha=0.2, label="95% Confidence Interval")
#plt.scatter(X_old, Y_imag_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()
#

def gp_surrogate_derivative(x_star):
    
    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)

    # Extract lengthscale parameter
    lengthscale = kernel.lengthscale
    variance = kernel.variance
    # Compute derivative of the RBF
    dkdx = -(x_star_norm - X) / (lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2))

    # Compute gradient
    dy_dx_norm = dkdx.T @ alpha

    # Convert back to original scale (denormalization)
    dy_dx = dy_dx_norm * (Y_std_old / X_std_old)

    return dy_dx.item()


def gp_surrogate_covariance_deriv(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old

    lengthscale = kernel.lengthscale
    variance = kernel.variance
    # Compute covariance vector k(X^*, X)


    k_star_deriv = -((x_star_norm-X) / lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2)) 
    k_star_star_deriv = variance / (lengthscale**2)
#    K = kernel.K(X,X)

    L = cholesky(K + np.eye(len(X)) * 1e-8, lower=True)

    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star_deriv)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star_deriv - k_star_deriv.T @ v
    y_surrogate_cov = y_surrogate_cov.item()
    # Compute gradient

    # Convert back to original scale (denormalization)

    yy_imag = y_surrogate_cov * (Y_std_old**2/X_std_old**2)

    return np.array(yy_imag).flatten()

Y_list = np.array([gp_surrogate_derivative(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list)


Y_list_cov = np.array([gp_surrogate_covariance_deriv(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list_cov)


std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
print(upper_bound_imag)
lower_bound = Y_list - 1.96 * std_dev
print(lower_bound_imag)
# Ensure Y_list is in correct shape
#Y_list = Y_list.flatten()  # Convert to 1D array if needed
#Y_list_cov = Y_list_cov.flatten()
# Plot the surrogate model predictions
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
#plt.scatter(X_old, Y_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()






def gp_surrogate_derivative_2(x_star):
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)

    # Extract lengthscale parameter
    lengthscale = kernel.lengthscale
    variance = kernel.variance
    # Compute derivative of the RBF
    d2kdx2 = ((x_star_norm - X)**2 / (lengthscale**4) - 1 / (lengthscale**2)) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2)) 
    # Compute gradient
    dy_dx_norm = d2kdx2.T @ alpha

    # Convert back to original scale (denormalization)
    dy_dx = dy_dx_norm * (Y_std_old / X_std_old**2)

    return dy_dx

def gp_surrogate_derivative_3(x_star):

    x_star_norm = (x_star - X_mean_old) / X_std_old                                                                   # Compute covariance vector k(X^*, X)
    k_star_imag = kernel.K(x_star_norm, X)
    # Constants
    lengthscale = kernel.lengthscale
    variance = kernel.variance

    # Compute difference
    diff = x_star_norm - X

    # Exponential term
    exp_term = np.exp(- (diff ** 2) / (2 * lengthscale ** 2))

    # Derivative
    d2kdx2_derivative = variance * exp_term * (3 * diff / lengthscale ** 4 - (diff ** 3) / lengthscale ** 6)



    d2y_dx2_norm = d2kdx2_derivative.T @ alpha

    # Convert back to original scale (denormalization)
    d2y_dx2 = d2y_dx2_norm * (Y_std_old / X_std_old**3)

    return d2y_dx2


X_test = np.array([1.2000E-03,
1.2500E-03,
1.3000E-03,
1.3500E-03,
1.4000E-03,
1.4500E-03,
1.5000E-03,
1.5500E-03,
1.6000E-03,
1.6500E-03,
1.7000E-03,
1.7500E-03,
1.8000E-03,
1.8500E-03]).reshape(-1,1)
#Y_derivative_list = []
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Y_derivative = gp_surrogate_derivative(x_star_reshaped)
#
#    # Append the result to the list
#    Y_derivative_list.append(Y_derivative)
#
## Convert the list to a numpy array for convenience (optional)
#Y_derivative_array = np.array(Y_derivative_list)
#
## Print the analytical derivatives
#print("Analytical derivatives:\n", Y_derivative_array)
#

Y_list = np.array([gp_surrogate_covariance_deriv(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list)



                                                                                                      
# Ensure Y_list is in correct shape
#Y_list = Y_list.flatten()  # Convert to 1D array if needed
## Plot the surrogate model predictions
#plt.figure(figsize=(8, 5))
#plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
##plt.scatter(X_old, Y_old, color='r', marker='x', label="Training Data")  # Original data points
#plt.xlabel("X_test (New Inputs)")
#plt.ylabel("Predicted Y")
#plt.title("Gaussian Process Surrogate Model")
#plt.legend()
#plt.grid()
#plt.show()
#

def gp_surrogate_derivative_imag(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star_imag = kernel_imag.K(x_star_norm, X)

    # Extract lengthscale parameter
    lengthscale_imag = kernel_imag.lengthscale
    variance_imag = kernel_imag.variance
    # Compute derivative of the RBF
    dkdx = -(x_star_norm - X) / (lengthscale_imag**2) * variance_imag * np.exp(-((x_star_norm - X)**2) / (2*lengthscale_imag**2))

    # Compute gradient
    dy_dx_norm = dkdx.T @ alpha_imag

    # Convert back to original scale (denormalization)
    dy_dx = dy_dx_norm * (Y_imag_std_old / X_std_old)

    return dy_dx.item()



def gp_surrogate_covariance_deriv_imag(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old

    lengthscale = kernel_imag.lengthscale
    variance = kernel_imag.variance
    # Compute covariance vector k(X^*, X)


    k_star_deriv = -((x_star_norm-X) / lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2)) 
    print("k_star_deriv", k_star_deriv.shape)
    k_star_star_deriv = variance / (lengthscale**2)
    print("k_star_star_deriv", k_star_star_deriv.shape)
    K = kernel_imag.K(X,X)

    L = cholesky(K + np.eye(len(X)) * 1e-10, lower=True)

    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star_deriv)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star_deriv - k_star_deriv.T @ v
    print("y_surrogate_cov shape:", y_surrogate_cov.shape) 
    y_surrogate_cov = y_surrogate_cov.item()
    # Compute gradient

    # Convert back to original scale (denormalization)

    yy_imag = y_surrogate_cov * (Y_imag_std_old**2/X_std_old**2)

    return np.array(yy_imag).flatten()

Y_list = np.array([gp_surrogate_covariance_deriv_imag(x.reshape(1, 1)) for x in new_points]).flatten()
print(Y_list)




# Ensure Y_list is in correct shape
Y_list = Y_list.flatten()  # Convert to 1D array if needed
# Plot the surrogate model predictions
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(X_old, Y_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()



def gp_surrogate_derivative_imag_2(x_star):
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star_imag = kernel_imag.K(x_star_norm, X)

    # Extract lengthscale parameter
    lengthscale_imag = kernel_imag.lengthscale
    variance_imag = kernel_imag.variance
    # Compute derivative of the RBF
    d2kdx2 = ((x_star_norm - X)**2 / (lengthscale_imag**4) - 1 / (lengthscale_imag**2)) * variance_imag * np.exp(-((x_star_norm - X)**2) / (2*lengthscale_imag**2))
    # Compute gradient
    dy_dx_norm = d2kdx2.T @ alpha_imag

    # Convert back to original scale (denormalization)
    dy_dx = dy_dx_norm * (Y_imag_std_old / X_std_old**2)

    return dy_dx

def gp_surrogate_derivative_imag_3(x_star):

    x_star_norm = (x_star - X_mean_old) / X_std_old                                                                   # Compute covariance vector k(X^*, X)
    k_star_imag = kernel_imag.K(x_star_norm, X)
    # Constants
    lengthscale_imag = kernel_imag.lengthscale
    variance_imag = kernel_imag.variance
    
    # Compute difference
    diff = x_star_norm - X
    
    # Exponential term
    exp_term = np.exp(- (diff ** 2) / (2 * lengthscale_imag ** 2))
    
    # Derivative
    d2kdx2_derivative = variance_imag * exp_term * (3 * diff / lengthscale_imag ** 4 - (diff ** 3) / lengthscale_imag ** 6)
    
    
    d2y_dx2_norm = d2kdx2_derivative.T @ alpha_imag

    # Convert back to original scale (denormalization)
    d2y_dx2 = d2y_dx2_norm * (Y_std_old / X_std_old**3)

    return d2y_dx2








X_test = np.array([1.2000E-03,
1.2500E-03,
1.3000E-03,
1.3500E-03,
1.4000E-03,
1.4500E-03,
1.5000E-03,
1.5500E-03,
1.6000E-03,
1.6500E-03,
1.7000E-03,
1.7500E-03,
1.8000E-03,
1.8500E-03]).reshape(-1,1)
Y_imag_derivative_list = []
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Y_derivative_imag = gp_surrogate_derivative_imag(x_star_reshaped)

    # Append the result to the list
    Y_imag_derivative_list.append(Y_derivative_imag)

# Convert the list to a numpy array for convenience (optional)
Y_derivative_array_imag = np.array(Y_imag_derivative_list)

# Print the analytical derivatives
print("Analytical derivatives:\n", Y_derivative_array_imag)

def gp_surrogate_velocity(x_star):
    # Normalize input
    x_star = np.atleast_2d(x_star).T
    velocity = x_star*np.sqrt(gp_surrogate_derivative_imag(x_star)**2+gp_surrogate_derivative(x_star)**2)

    return velocity.item()

def gp_surrogate_velocity_variance(x_star):
    # Normalize input
    x_star = np.atleast_2d(x_star).T
    velocity = (x_star**2)*((gp_surrogate_derivative(x_star))**2*gp_surrogate_covariance_deriv(x_star)+(gp_surrogate_derivative_imag(x_star))**2*gp_surrogate_covariance_deriv_imag(x_star))/((gp_surrogate_derivative(x_star))**2+(gp_surrogate_derivative_imag(x_star))**2)

    return velocity.item()


X_test = np.array([1.2000E-03,
1.2500E-03,
1.3000E-03,
1.3500E-03,
1.4000E-03,
1.4500E-03,
1.5000E-03,
1.5500E-03,
1.6000E-03,
1.6500E-03,
1.7000E-03,
1.7500E-03,
1.8000E-03,
1.8500E-03,
2.000E-03,
2.300E-03,
2.700E-03,
2.800E-03                   ]).reshape(-1,1)
Velocity_list = []
Velocity_variance_list = []
x_min = np.min(X_old)
x_max = np.max(X_old)

# Create an array of 100 evenly spaced points between x_min and x_max
new_points = np.linspace(x_min, x_max, 300).reshape(-1, 1)

for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = gp_surrogate_velocity(x_star_reshaped)
    Velocity_variance = gp_surrogate_velocity_variance(x_star_reshaped)
    # Append the result to the list
    Velocity_list.append(Velocity)
    Velocity_variance_list.append(Velocity_variance)
# Convert the list to a numpy array for convenience (optional)
Velocity_array = np.array(Velocity_list)
Velocity_var_array = np.array(Velocity_variance_list)
print("Velocity_variance:\n", Velocity_var_array)
std_dev = np.sqrt(Velocity_var_array)
upper_bound = Velocity_array + 1.96 * std_dev
lower_bound = Velocity_array - 1.96 * std_dev
print("Velocity:\n", Velocity_array)


#plt.figure(figsize=(8, 5))
#plt.plot(new_points.flatten(), Velocity_var_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Velocity_var_array.flatten(), label="GP Surrogate Model", color='b')
#plt.xlabel("X_test (New Inputs)")
#plt.ylabel("Predicted Y")
#plt.title("Gaussian Process Surrogate Model")
#plt.legend()
#plt.grid()
#plt.show()   
#
Velocity_deriv_list_old=[]
for x_star in X_old:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = gp_surrogate_velocity(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv_list_old.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_array_old = np.array(Velocity_deriv_list_old)



plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Velocity_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Velocity_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(X_old, Velocity_array_old.flatten(), color='r', marker='x', label="Training Data")  # Original data points
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()


def velocity_deriv(x_scalar):

    x = np.atleast_1d(x_scalar)
    x = x.reshape(-1,1)
    velocity_deriv = (gp_surrogate_derivative_imag(x)**2+gp_surrogate_derivative(x)**2) + x*(gp_surrogate_derivative_imag(x)*gp_surrogate_derivative_imag_2(x)+gp_surrogate_derivative(x)*gp_surrogate_derivative_2(x))
    return velocity_deriv


x_min = np.min(X_old)
x_max = np.max(X_old)

# Create an array of 100 evenly spaced points between x_min and x_max
new_points = np.linspace(x_min, x_max, 300).reshape(-1, 1)
Velocity_deriv_list = []
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = velocity_deriv(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv_list.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_deriv_array = np.array(Velocity_deriv_list)


Velocity_deriv_list_old = []
for x_star in X_old:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = velocity_deriv(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv_list_old.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_deriv_array_old = np.array(Velocity_deriv_list_old)


# Create an array of 100 evenly spaced points between x_min and x_max
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Velocity_deriv_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Velocity_deriv_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(X_old, Velocity_deriv_array_old.flatten(), color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()






Velocity_deriv_list = []
for x_star in X_old:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = velocity_deriv(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv_list.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_deriv_array = np.array(Velocity_deriv_list)

# Create an array of 100 evenly spaced points between x_min and x_max
plt.figure(figsize=(8, 5))
plt.plot(X_old.flatten(), Velocity_deriv_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(X_old.flatten(), Velocity_deriv_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(X_old, Y_imag_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()





#print("Velocity:\n", Velocity_deriv_array)

def velocity_deriv_derivative(x_scalar):
    x = np.atleast_1d(x_scalar)
    x = x.reshape(-1, 1)

    gp_i_prime = gp_surrogate_derivative_imag(x)
    gp_i_double_prime = gp_surrogate_derivative_imag_2(x)
    gp_i_triple_prime = gp_surrogate_derivative_imag_3(x)

    gp_prime = gp_surrogate_derivative(x)
    gp_double_prime = gp_surrogate_derivative_2(x)
    gp_triple_prime = gp_surrogate_derivative_3(x)

    velocity_deriv_prime = (
        2 * gp_i_prime * gp_i_double_prime +
        2 * gp_prime * gp_double_prime +
        x * (gp_i_prime * gp_i_triple_prime + gp_i_double_prime ** 2 +
             gp_prime * gp_triple_prime + gp_double_prime ** 2) +
        gp_i_prime * gp_i_double_prime +
        gp_prime * gp_double_prime
    )

    return velocity_deriv_prime


Velocity_deriv2_list = []
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = velocity_deriv_derivative(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv2_list.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_deriv2_array = np.array(Velocity_deriv2_list)

plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Velocity_deriv2_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Velocity_deriv2_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(X_old, Y_imag_old, color='r', marker='x', label="Training Data")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Gaussian Process Surrogate Model")
plt.legend()
plt.grid()
plt.show()



from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

new_points = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
# Convert X_test to intervals between consecutive points
intervals = [(new_points[i], new_points[i+1]) for i in range(len(new_points) - 1)]
# Find roots in each interval
roots = []
energy = []
imag_energy = []
deriv = []
imag_deriv = []
for interval in intervals:
    try:
        root_result = root_scalar(velocity_deriv, bracket=interval, method='bisect')
        print (root_result.root)
        print(velocity_deriv_derivative(root_result.root))
        if root_result.converged and velocity_deriv_derivative(root_result.root) > 0:  # Ensure the solver found a valid root
            roots.append(root_result.root)
            energy.append(gp_surrogate_model(root_result.root))
            imag_energy.append(gp_surrogate_model_imag(root_result.root))
            deriv.append(gp_surrogate_derivative(root_result.root))
            imag_deriv.append(gp_surrogate_derivative_imag(root_result.root))
    except ValueError:
        # No root found in this interval (e.g., function does not change sign)
        pass


np.savetxt("Minima.dat",np.column_stack((roots, energy, imag_energy, deriv, imag_deriv)), fmt="%.10f")
#print("Roots:", roots)


x_min, x_max = X_old[0], X_old[len(X_old)-1]   # Example interval

# Find the maximum by minimizing the negative function
result = minimize_scalar(lambda x: -gp_surrogate_covariance(x), bounds=(x_min, x_max), method='bounded')

# Extract the maximum point
x_max_value = result.x
y_max_value = gp_surrogate_covariance(result.x)  # Negate back to original function value

x_max_value = x_max_value.item()
y_max_value = y_max_value.item()


# Print the results
np.savetxt("Max_Uncertainty.dat",np.column_stack((x_max_value,y_max_value)), fmt="%.10f")



#intervals = [(0.0012, 0.0019)]
#
## Find roots in different intervals
#roots = [root_scalar(velocity_deriv, bracket=interval, method='bisect').root for interval in intervals]
#print("Roots:", roots)
#
#




#from sympy import real_roots, symbols
#
#x = symbols('x')
#
#f = x*np.sqrt(gp_surrogate_derivative_imag(x)**2+gp_surrogate_derivative(x)**2)
#d1 = f.diff(x)
#d2 = d1.diff(x)  # = f.diff(x,2)
#extrema = real_roots(d1)
#for i in extrema:
#  if d2.subs(x, i).is_positive:
#     print('minimum',i)
#  else:
#     print('maxima',i)
