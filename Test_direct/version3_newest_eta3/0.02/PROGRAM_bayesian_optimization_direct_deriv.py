import numpy as np
import matplotlib.pyplot as plt
import GPy

np.random.seed(42)

# Loading data from training set and big scan from QP to compare

data = np.loadtxt('datafile.dat')
data_big = np.loadtxt('datafile_big_scan.dat')

X = data[:,0].reshape(-1,1)
Y = data[:,1].reshape(-1,1)
Y_imag = data[:,2].reshape(-1,1)
Y_deriv_scaled_old = data[:,3].reshape(-1,1)
Y_imag_deriv_scaled_old = data[:,4].reshape(-1,1)
Y_deriv_old = data[:,5].reshape(-1,1)
Y_imag_deriv_old = data[:,6].reshape(-1,1)
velocity_old = data[:,7].reshape(-1,1)

X_big = data_big[:,0].reshape(-1,1)
Y_big = data_big[:,1].reshape(-1,1)
Y_imag_big = data_big[:,2].reshape(-1,1)
Y_deriv_scaled_big = data_big[:,3].reshape(-1,1)
Y_imag_deriv_scaled_big = data_big[:,4].reshape(-1,1)
Y_deriv_big = data_big[:,5].reshape(-1,1)
Y_imag_deriv_big = data_big[:,6].reshape(-1,1)
velocity_big = data_big[:,7].reshape(-1,1)



#Defining kernels for the regressions for the real and imaginary energy

kernel = GPy.kern.RBF(input_dim=1, variance=np.random.uniform(0.5, 4), lengthscale=np.random.uniform(0.5, 4))
kernel_imag = GPy.kern.RBF(input_dim=1, variance=np.random.uniform(0.5, 4), lengthscale=np.random.uniform(0.5, 4))

# Setting priors to guide the optimization of the hyperparameters
kernel_imag.variance.set_prior(GPy.priors.Gamma(2., 0.5))  # Example, set prior explicitly
kernel_imag.lengthscale.set_prior(GPy.priors.Gamma(2., 0.5))
kernel.variance.set_prior(GPy.priors.Gamma(2., 0.5))  # Example, set prior explicitly
kernel.lengthscale.set_prior(GPy.priors.Gamma(2., 0.5))


#Normalization of the training data
X_mean_old = np.mean(X)
velocity_mean_old = np.mean(velocity_old)
X_std_old = np.std(X)
velocity_std_old = np.std(velocity_old)

X_old=X
velocity=velocity_old

X = (X - np.mean(X)) / np.std(X)
velocity = (velocity - np.mean(velocity)) / np.std(velocity)


#Optimization of the hyperparameters + fixing the noise to 0
m = GPy.models.GPRegression(X,velocity,kernel)
m.Gaussian_noise.variance.fix(0.0)

#m.likelihood.variance = 1e-1
from IPython.display import display

#m.optimize(messages=True)
m.optimize_restarts(messages=False, max_iters=1000, optimizer='bfgs', num_restarts=40)
display(m)
fig = m.plot()
print("Noise variance:", m.Gaussian_noise.variance.values)
plt.show()
#
#Defining the number of points used in the plots of the Surrogate Model
x_min = np.min(X)
x_max = np.max(X)
new_points = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
new_points2 = np.linspace(x_min, x_max, 10000).reshape(-1, 1)

K = kernel.K(X,X)
K_imag = kernel_imag.K(X,X)


from scipy.linalg import cho_solve, cholesky

# Compute Cholesky decomposition
L = cholesky(K + np.eye(len(X)) * 1e-11, lower=True)  # Add jitter for stability

# Solve for alpha (K * alpha = Y)
alpha = cho_solve((L, True), velocity)

def gp_surrogate_model(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)

    # Extract lengthscale parameter
    # Compute gradient
    velocity_surrogate = k_star @ alpha

    # Convert back to original scale (denormalization)
    velocity_surrogate = velocity_surrogate * velocity_std_old + velocity_mean_old

    return velocity_surrogate.flatten()


def gp_surrogate_covariance(x_star):

    x_star = np.atleast_1d(x_star)
    x_star = x_star.reshape(-1,1)
    # Normalize input
    x_star_norm = (x_star - X_mean_old) / X_std_old
    # Compute covariance vector k(X^*, X)
    k_star = kernel.K(x_star_norm, X)
    k_star_star = kernel.K(x_star_norm, x_star_norm)

    L = cholesky(K + np.eye(len(X)) * 1e-11, lower=True)
    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star.T)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star - k_star @ v
    # Extract lengthscale parameter

    # Convert back to original scale (denormalization)
    yy = y_surrogate_cov * velocity_std_old**2
    return yy.flatten()

## Plotting the surrogate model for the real and the imaginary energies

x_min = np.min(X_old)
x_max = np.max(X_old)
new_points = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
new_points2 = np.linspace(x_min, x_max, 10000).reshape(-1, 1)


Y_list = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in new_points]).flatten()
Y_list_lines = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in new_points2]).flatten()
Y_list_training = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in X_old]).flatten()

Y_list_cov = np.array([gp_surrogate_covariance(x.reshape(1, 1)) for x in new_points]).flatten()

std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev

# Ensure Y_list is in correct shape
#Y_list = Y_list.flatten()  # Convert to 1D array if needed
#Y_list_cov = Y_list_cov.flatten()
# Plot the surrogate model predictions for the real and imaginary energies
plt.figure(figsize=(8, 5))
plt.plot(new_points2.flatten(), Y_list_lines.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b',s=10)
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
plt.scatter(X_old, Y_list_training.flatten(), color='g', marker='x', label="Regression at training point",s=70)  # Original data points
plt.scatter(X_old, velocity_old, color='r', marker='x', label="Training Data",s=70)  # Original data points
plt.scatter(X_big, velocity_big, color='k', marker='x', label="Big Scan",s=10)
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Velocity Gaussian Process")
plt.legend()
plt.grid()
plt.show()
#

# Calcualting derivatives (and variances) of the Surrogate Model for teh Real energy

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
    dy_dx = dy_dx_norm * (velocity_std_old / X_std_old)

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

    L = cholesky(K + np.eye(len(X)) * 1e-11, lower=True)

    # Compute the covariance
    # Using cho_solve to ensure numerical stability
    v = cho_solve((L, True), k_star_deriv)

    # Compute the covariance of the surrogate
    y_surrogate_cov = k_star_star_deriv - k_star_deriv.T @ v
    y_surrogate_cov = y_surrogate_cov.item()
    # Compute gradient

    # Convert back to original scale (denormalization)

    yy_imag = y_surrogate_cov * (velocity_std_old**2/X_std_old**2)

    return np.array(yy_imag).flatten()


#Plotting the 1st derivative with the correspoind Uncertainty
Y_list = np.array([gp_surrogate_derivative(x.reshape(1, 1)) for x in new_points]).flatten()
Y_list_training = np.array([gp_surrogate_derivative(x.reshape(1, 1)) for x in X_old]).flatten()

Y_list_cov = np.array([gp_surrogate_covariance_deriv(x.reshape(1, 1)) for x in new_points]).flatten()

std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev

# Plot the surrogate model predictions
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
plt.scatter(X_old, Y_list_training.flatten(), color='g', marker='x', label="Regression at training point")  # Original data points
plt.scatter(X_old, Y_deriv_old, color='r', marker='x', label="QP data")  # Original data points
plt.scatter(X_big, Y_deriv_big, color='k', marker='x', label="QP Big Scan")  # Original data points
plt.xlabel("X_test (New Inputs)")
plt.ylabel("Predicted Y")
plt.title("Real Derivative Surrogate Model")
plt.legend()
plt.grid()
plt.show()

#
# Calculating higher order derivatives (necessary to find the minimum of the velocity)

#Computing the velocity with the corresponding uncertanty


def gp_surrogate_velocity_negative(x):
    # Normalize input
    x = np.atleast_2d(x).T
    velocity = -gp_surrogate_model(x)
    return velocity

from matplotlib import gridspec

#
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def EI_acquisition(x, Max_Y,xi_old):
    """Compute the Expected Improvement (EI) acquisition function."""
    mean = gp_surrogate_velocity_negative(x)  # Assuming negative velocity is being maximized
    std = np.sqrt(gp_surrogate_covariance(x))  # Use the known variance expression
    xi = xi_old * std

    if std < 1e-11:  # Avoid division by zero
        return max(mean - np.max(Max_Y) + xi, 0)

    f_best = np.max(Max_Y)  # Best observed function value (minimization)
    
    improvement = mean - f_best + xi  # Difference from best observed
    Z = improvement / std  # Standardized improvement
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)  # EI formula
    
    return max(ei, 0)  # Ensure EI is non-negative


def LB_acquisition(kappa, x):
    """Compute the UCB acquisition function using the known function and variance."""
    mean = gp_surrogate_model(x)  # Assuming negative velocity is being maximized
    std = np.sqrt(gp_surrogate_covariance(x))  # Use the known variance expression
    return mean - kappa * std  # UCB formula


#

def gp_surrogate_derivative2(x_star):
    
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
    dy_dx_norm = dy_dx_norm*(velocity_std_old/X_std_old)
    # Convert back to original scale (denormalization)
#    dy_dx = dy_dx_norm * (velocity_std_old / X_std_old)

    return dy_dx_norm.item()
def gp_surrogate_covariance_deriv2(x_star):
    x_star = np.atleast_1d(x_star).reshape(-1, 1)
    x_star_norm = (x_star - X_mean_old) / X_std_old

    lengthscale = kernel.lengthscale
    variance = kernel.variance

    # Compute the derivative of k(x*, X)
    k_star_deriv = -(x_star_norm - X) / (lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2 * lengthscale**2))
    # Compute the derivative of k(x*, x*) correctly
#    k_star_star_deriv = (1 / lengthscale**2) * variance
    k_star = kernel.K(x_star_norm, X)

    # Compute Cholesky decomposition for numerical stability
    K = kernel.K(X, X)
    L = cholesky(K + np.eye(len(X)) * 1e-8, lower=True)
    # Solve for v
    v = cho_solve((L, True), k_star_deriv)

    # Compute the derivative of the standard deviation
    gp_cov = gp_surrogate_covariance(x_star) / (velocity_std_old**2)
    y_surrogate_cov = (1/np.sqrt(gp_cov)) * (-k_star @ v)
    y_surrogate_cov = y_surrogate_cov * (velocity_std_old/X_std_old)
    return y_surrogate_cov.flatten()

#def gp_surrogate_covariance_deriv2(x_star):
#
#    x_star = np.atleast_1d(x_star)
#    x_star = x_star.reshape(-1,1)
#    # Normalize input
#    x_star_norm = (x_star - X_mean_old) / X_std_old
#
#    lengthscale = kernel.lengthscale
#    variance = kernel.variance
#    # Compute covariance vector k(X^*, X)
#
#
#    k_star_deriv = -(x_star_norm-X) / (lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2))
#    k_star_star_deriv = (x_star_norm-x_star_norm) / (lengthscale**2) * variance * np.exp(-((x_star_norm - x_star_norm)**2) / (2*lengthscale**2))
#    K = kernel.K(X,X)
#
#    L = cholesky(K + np.eye(len(X)) * 1e-11, lower=True)
#
#    # Compute the covariance
#    # Using cho_solve to ensure numerical stability
#    v = cho_solve((L, True), k_star_deriv)
#
#    # Compute the covariance of the surrogate
#    y_surrogate_cov = (1/(2*np.sqrt(gp_surrogate_covariance(x_star)/(velocity_std_old**2)))) *(k_star_star_deriv - 2*k_star_deriv.T @ v)
#    y_surrogate_cov = y_surrogate_cov.item()
#    # Compute gradient
#
#    # Convert back to original scale (denormalization)
#
##    yy_imag = y_surrogate_cov * (velocity_std_old/X_std_old)
#
#    return np.array(y_surrogate_cov).flatten()
#

def d_LB_acquisition(kappa, x):
    """Compute the derivative of the LB acquisition function."""
    x = np.atleast_1d(x).reshape(-1, 1)
   
    func = gp_surrogate_derivative2(x) - kappa* gp_surrogate_covariance_deriv2(x)
    # Normalize input
    return func






#Plotting the velocity
Velocity_list = []
Velocity_variance_list = []

for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = gp_surrogate_model(x_star_reshaped)
    Velocity_variance = gp_surrogate_covariance(x_star_reshaped)
    # Append the result to the list
    Velocity_list.append(Velocity)
    Velocity_variance_list.append(Velocity_variance)
# Convert the list to a numpy array for convenience (optional)
Velocity_array = np.array(Velocity_list)
Velocity_var_array = np.array(Velocity_variance_list)
std_dev = np.sqrt(Velocity_var_array)
upper_bound = Velocity_array + 1.96 * std_dev
lower_bound = Velocity_array - 1.96 * std_dev

#
Velocity_deriv_list_old=[]
for x_star in X_old:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Velocity = gp_surrogate_model(x_star_reshaped)

    # Append the result to the list
    Velocity_deriv_list_old.append(Velocity)

# Convert the list to a numpy array for convenience (optional)
Velocity_array_old = np.array(Velocity_deriv_list_old)

Acquisition_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Acquisition = LB_acquisition(3.0,x_star_reshaped)

    # Append the result to the list
    Acquisition_list.append(Acquisition)

# Convert the list to a numpy array for convenience (optional)
Acquisition_array = np.array(Acquisition_list)

Acquisition_EI_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Acquisition_EI = EI_acquisition(x_star_reshaped, -Velocity_array_old,2.0)

    # Append the result to the list
    Acquisition_EI_list.append(Acquisition_EI)

# Convert the list to a numpy array for convenience (optional)
Acquisition_EI_array = np.array(Acquisition_EI_list)

#print(Acquisition_EI_array)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Primary plot (GP Surrogate Model and other data points)
ax1.plot(new_points.flatten(), Velocity_array.flatten(), label="GP Surrogate Model", color='b')
ax1.scatter(new_points.flatten(), Velocity_array.flatten(), color='b')
ax1.scatter(X_old, Velocity_array_old.flatten(), color='g', marker='x', label="Regression in training points")
ax1.scatter(X_old, velocity_old, color='r', marker='x', label="QP data")
ax1.scatter(X_big, velocity_big, color='k', marker='x', label="QP Big Scan")
ax1.scatter(new_points, Acquisition_array, color='y', marker='o', label="LB Acquisition Function")
ax1.fill_between(new_points.flatten(), lower_bound.flatten(), upper_bound.flatten(), color='b', alpha=0.2, label="95% Confidence Interval")

# Labels and title for main plot
ax1.set_ylabel("Predicted Y / Acquisition Values")
ax1.set_title("Velocity Surrogate Model")
ax1.legend(loc="upper left")
ax1.grid()

# Secondary plot (EI Acquisition Function)
ax2.scatter(new_points, Acquisition_EI_array, color='m', marker='o', label="EI Acquisition Function")
ax2.set_xlabel("X_test (New Inputs)")
ax2.set_ylabel("EI Value")
ax2.legend(loc="upper left")
ax2.grid()

plt.tight_layout()
plt.show()



#new_points = np.linspace(x_min, x_max, 10000).reshape(-1, 1)
Acquisition_LB_deriv_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Acquisition_LB_deriv = d_LB_acquisition(3.0,x_star_reshaped)

    # Append the result to the list
    Acquisition_LB_deriv_list.append(Acquisition_LB_deriv)

# Convert the list to a numpy array for convenience (optional)
Acquisition_LB_deriv_array = np.array(Acquisition_LB_deriv_list)


Mean_GP_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Mean_GP = gp_surrogate_model(x_star_reshaped)

    # Append the result to the list
    Mean_GP_list.append(Mean_GP)

# Convert the list to a numpy array for convenience (optional)
Mean_GP_array = np.array(Mean_GP_list)


Mean_GP_deriv_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Mean_GP_deriv = gp_surrogate_derivative2(x_star_reshaped)

    # Append the result to the list
    Mean_GP_deriv_list.append(Mean_GP_deriv)

# Convert the list to a numpy array for convenience (optional)
Mean_GP_deriv_array = np.array(Mean_GP_deriv_list)

Std_GP_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Std_GP = gp_surrogate_covariance(x_star_reshaped)

    # Append the result to the list
    Std_GP_list.append(Std_GP)

# Convert the list to a numpy array for convenience (optional)
Std_GP_array = np.array(Std_GP_list)


Std_GP_deriv_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Std_GP_deriv = gp_surrogate_covariance_deriv2(x_star_reshaped)

    # Append the result to the list
    Std_GP_deriv_list.append(Std_GP_deriv)

# Convert the list to a numpy array for convenience (optional)
Std_GP_deriv_array = np.array(Std_GP_deriv_list)


plt.figure(figsize=(8, 5))
plt.title("Mean and derivative")
plt.plot(new_points.flatten(), Mean_GP_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Mean_GP_array.flatten(), label="GP Surrogate Model", color='b')


plt.figure(figsize=(8, 5))
plt.title("Mean and derivative")
plt.plot(new_points.flatten(), Mean_GP_deriv_array.flatten(), label="GP Surrogate Model", color='r')
plt.scatter(new_points.flatten(), Mean_GP_deriv_array.flatten(), color='r')
plt.show()


plt.figure(figsize=(8, 5))
plt.title("Std and derivative")
plt.plot(new_points.flatten(), Std_GP_array.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Std_GP_array.flatten(), label="GP Surrogate Model", color='b')

plt.figure(figsize=(8, 5))
plt.title("Std and derivative")
plt.plot(new_points.flatten(), Std_GP_deriv_array.flatten(), label="GP Surrogate Model", color='r')
plt.scatter(new_points.flatten(), Std_GP_deriv_array.flatten(), color='r')

plt.show()





Y_list = np.array([gp_surrogate_derivative(x.reshape(1, 1)) for x in new_points]).flatten()

Y_list_cov = np.array([gp_surrogate_covariance_deriv(x.reshape(1, 1)) for x in new_points]).flatten()

std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev

# Plot the surrogate model predictions
plt.figure(figsize=(8, 5))
plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
plt.plot(new_points.flatten(), Acquisition_LB_deriv_array.flatten(), label="GP Surrogate Model", color='r')
plt.scatter(new_points.flatten(), Acquisition_LB_deriv_array.flatten(), color='r')

# Labels and title for main plot

# Secondary plot (EI Acquisition Function)

plt.show()
#
# Calculating and plotting 1st and 2nd derivative of the velocity 
#
# Computing the minimas of the surrogate model (between the first and last training points)
#


bounds = [(x_min, x_max)]
kappa = 2.0
xi = 2.0
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar


def wrapper(x):
    return d_LB_acquisition(3.0,x)
#new_points = np.linspace(x_min, x_max, 10000).reshape(-1, 1)
intervals = [(new_points[i], new_points[i+1]) for i in range(len(new_points) - 1)]
# Find roots in each interval
roots = []
energy = []
imag_energy = []
deriv = []
imag_deriv = []

for interval in intervals:
    try:
        root_result = root_scalar(wrapper, bracket=interval, method='bisect')
        print(root_result.root)
        print(d_LB_acquisition(2.0,root_result.root))
        # print(velocity_deriv_derivative(root_result.root))

#        if root_result.converged and wrapper(root_result.root) > 0:  # Ensure the solver found a valid root
        roots.append(root_result.root)
            # energy.append(gp_surrogate_model(root_result.root))
            # imag_energy.append(gp_surrogate_model_imag(root_result.root))
            # deriv.append(gp_surrogate_derivative(root_result.root))
            # imag_deriv.append(gp_surrogate_derivative_imag(root_result.root))

    except ValueError:
        # No root found in this interval (e.g., function does not change sign)
        pass

best_root = None
best_lb_value = float("inf")  # Initialize with a high value
print(roots)
for root in roots:
    lb_value = LB_acquisition(kappa, np.atleast_2d(root))
    print(f"Root: {root}, LB_acquisition: {lb_value}")

    if lb_value < best_lb_value:
        best_root = root
        best_lb_value = lb_value

print(f"Best Root: {best_root}, Lowest LB_acquisition: {best_lb_value}")
data = np.column_stack((best_root, best_lb_value))  # Combine into a 2D array
np.savetxt("Optimal_LB", data, fmt="%.10f")

lbfgs_options = {
    "gtol": 1e-7,   # Stricter gradient tolerance (default is 1e-5)
    "ftol": 1e-10,  # Stricter function tolerance (default is ~2.2e-9)
    "maxiter": 20000  # Increase max iterations if needed
}

def multi_start_minimize_LB(n_starts):
    """Find the global minimum of the LB acquisition function using Multi-Start L-BFGS-B."""
    best_x = None
    best_fun = float("inf")  # Initialize with a very high value
    
    for _ in range(n_starts):
        x0 = np.random.uniform(x_min, x_max, size=(1,))  # Random initial point
        result = minimize(
            lambda x: LB_acquisition(kappa, np.atleast_2d(x)),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=lbfgs_options
        )
        if result.fun < best_fun:  # Keep track of the best result
            best_x, best_fun = result.x, result.fun
    
    return best_x, best_fun

def multi_start_maximize_EI(n_starts):
    """Find the global maximum of the EI acquisition function using Multi-Start L-BFGS-B."""
    best_x = None
    best_fun = -float("inf")  # Initialize with a very low value
    
    for _ in range(n_starts):
        x0 = np.random.uniform(x_min, x_max, size=(1,))  # Random initial point
        result = minimize(
            lambda x: -EI_acquisition(np.atleast_2d(x), -Velocity_array_old,xi),  # Negate EI for maximization
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=lbfgs_options
        )
        if -result.fun > best_fun:  # Keep track of the best result
            best_x, best_fun = result.x, -result.fun  # Revert negation
    
    return best_x, best_fun

# Run Multi-Start L-BFGS-B
x_min_LB, min_LB_value = multi_start_minimize_LB(n_starts=100)
print(f"Global minimum of LB acquisition at x = {x_min_LB}, value = {min_LB_value}")

x_max_EI, max_EI_value = multi_start_maximize_EI(n_starts=1000)
print(f"Global maximum of EI acquisition at x = {x_max_EI}, value = {max_EI_value}")
#def minimize_LB():
#    """Find the minimum of the LB acquisition function."""
#    result = minimize(
#        lambda x: LB_acquisition(kappa, np.atleast_2d(x)),  # Function to minimize
#        x0=np.random.uniform(x_min, x_max, size=(1,)),  # Initial guess
#        bounds=bounds,
#        method="L-BFGS-B"  # Suitable for smooth functions
#    )
#    return result.x, result.fun  # Return the x that minimizes LB and its value
#
#x_min_LB, min_LB_value = minimize_LB()
#print(f"Minimum of LB acquisition at x = {x_min_LB}, value = {min_LB_value}")
#
#
#def maximize_EI():
#    """Find the maximum of the EI acquisition function."""
#    result = minimize(
#        lambda x: -EI_acquisition(np.atleast_2d(x), -Velocity_array_old),  # Negate EI for maximization
#        x0=np.random.uniform(x_min, x_max, size=(1,)),  # Initial guess
#        bounds=bounds,
#        method="L-BFGS-B"
#    )
#    return result.x, -result.fun  # Return the x that maximizes EI and its value
#
#x_max_EI, max_EI_value = maximize_EI()
#print(f"Maximum of EI acquisition at x = {x_max_EI}, value = {max_EI_value}")
#

#data = np.column_stack((x_min_LB, gp_surrogate_model(x_min_LB)))  # Combine into a 2D array
#np.savetxt("Optimal_LB", data, fmt="%.10f")
#data = np.column_stack((x_max_EI, gp_surrogate_model(x_max_EI)))  # Combine into a 2D array
#np.savetxt("Optimal_EI", data, fmt="%.10f")

