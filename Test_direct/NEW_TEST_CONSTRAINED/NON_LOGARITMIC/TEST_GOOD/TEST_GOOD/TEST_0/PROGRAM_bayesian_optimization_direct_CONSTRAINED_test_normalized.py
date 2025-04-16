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
kernel.variance.set_prior(GPy.priors.LogGaussian(0., 0.5))  # Example, set prior explicitly
kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0., 0.5))


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
m.optimize_restarts(verbose=False,messages=False, max_iters=1000, optimizer='bfgs', num_restarts=40)
#display(m)
fig = m.plot()
print("Noise variance:", m.Gaussian_noise.variance.values)
#plt.show()
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
#plt.figure(figsize=(8, 5))
#plt.plot(new_points2.flatten(), Y_list_lines.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b',s=10)
#plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
#plt.scatter(X_old, Y_list_training.flatten(), color='g', marker='x', label="Regression at training point",s=70)  # Original data points
#plt.scatter(X_old, velocity_old, color='r', marker='x', label="Training Data",s=70)  # Original data points
#plt.scatter(X_big, velocity_big, color='k', marker='x', label="Big Scan",s=10)
#plt.xlabel("X_test (New Inputs)")
#plt.ylabel("Predicted Y")
#plt.title("Velocity Gaussian Process")
#plt.legend()
#plt.grid()
#plt.show()
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
#plt.figure(figsize=(8, 5))
#plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
#plt.scatter(X_old, Y_list_training.flatten(), color='g', marker='x', label="Regression at training point")  # Original data points
#plt.scatter(X_old, Y_deriv_old, color='r', marker='x', label="QP data")  # Original data points
#plt.scatter(X_big, Y_deriv_big, color='k', marker='x', label="QP Big Scan")  # Original data points
#plt.xlabel("X_test (New Inputs)")
#plt.ylabel("Predicted Y")
#plt.title("Real Derivative Surrogate Model")
#plt.legend()
#plt.grid()
#plt.show()
#
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

#def gp_surrogate_derivative2(x_star):
#    
#    x_star = np.atleast_1d(x_star)
#    x_star = x_star.reshape(-1,1)
#    # Normalize input
#    x_star_norm = (x_star - X_mean_old) / X_std_old
#    # Compute covariance vector k(X^*, X)
#    k_star = kernel.K(x_star_norm, X)
#
#    # Extract lengthscale parameter
#    lengthscale = kernel.lengthscale
#    variance = kernel.variance
#    # Compute derivative of the RBF
#    dkdx = -(x_star_norm - X) / (lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2*lengthscale**2))
#
#    # Compute gradient
#    dy_dx_norm = dkdx.T @ alpha
#    dy_dx_norm = dy_dx_norm*(velocity_std_old/X_std_old)
#    # Convert back to original scale (denormalization)
##    dy_dx = dy_dx_norm * (velocity_std_old / X_std_old)
#
#    return dy_dx_norm.item()
#def gp_surrogate_covariance_deriv2(x_star):
#    x_star = np.atleast_1d(x_star).reshape(-1, 1)
#    x_star_norm = (x_star - X_mean_old) / X_std_old
#
#    lengthscale = kernel.lengthscale
#    variance = kernel.variance
#
#    # Compute the derivative of k(x*, X)
#    k_star_deriv = -(x_star_norm - X) / (lengthscale**2) * variance * np.exp(-((x_star_norm - X)**2) / (2 * lengthscale**2))
#    # Compute the derivative of k(x*, x*) correctly
##    k_star_star_deriv = (1 / lengthscale**2) * variance
#    k_star = kernel.K(x_star_norm, X)
#
#    # Compute Cholesky decomposition for numerical stability
#    K = kernel.K(X, X)
#    L = cholesky(K + np.eye(len(X)) * 1e-8, lower=True)
#    # Solve for v
#    v = cho_solve((L, True), k_star_deriv)
#
#    # Compute the derivative of the standard deviation
#    gp_cov = gp_surrogate_covariance(x_star) / (velocity_std_old**2)
#    y_surrogate_cov = (1/np.sqrt(gp_cov)) * (-k_star @ v)
#    y_surrogate_cov = y_surrogate_cov * (velocity_std_old/X_std_old)
#    return y_surrogate_cov.flatten()


#def d_LB_acquisition(kappa, x):
#    """Compute the derivative of the LB acquisition function."""
#    x = np.atleast_1d(x).reshape(-1, 1)
#   
#    func = gp_surrogate_derivative2(x) - kappa* gp_surrogate_covariance_deriv2(x)
#    # Normalize input
#    return func
#

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
    Acquisition = LB_acquisition(1.0,x_star_reshaped)

    # Append the result to the list
    Acquisition_list.append(Acquisition)

# Convert the list to a numpy array for convenience (optional)
Acquisition_array = np.array(Acquisition_list)

Acquisition_EI_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Acquisition_EI = EI_acquisition(x_star_reshaped, -Velocity_array_old,1.0)

    # Append the result to the list
    Acquisition_EI_list.append(Acquisition_EI)

# Convert the list to a numpy array for convenience (optional)
Acquisition_EI_array = np.array(Acquisition_EI_list)

#print(Acquisition_EI_array)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Primary plot (GP Surrogate Model and other data points)
#ax1.plot(new_points.flatten(), Velocity_array.flatten(), label="GP Surrogate Model", color='b')
#ax1.scatter(new_points.flatten(), Velocity_array.flatten(), color='b')
#ax1.scatter(X_old, Velocity_array_old.flatten(), color='g', marker='x', label="Regression in training points")
#ax1.scatter(X_old, velocity_old, color='r', marker='x', label="QP data")
#ax1.scatter(X_big, velocity_big, color='k', marker='x', label="QP Big Scan")
#ax1.scatter(new_points, Acquisition_array, color='y', marker='o', label="LB Acquisition Function")
#ax1.fill_between(new_points.flatten(), lower_bound.flatten(), upper_bound.flatten(), color='b', alpha=0.2, label="95% Confidence Interval")
#
## Labels and title for main plot
#ax1.set_ylabel("Predicted Y / Acquisition Values")
#ax1.set_title("Velocity Surrogate Model")
#ax1.legend(loc="upper left")
#ax1.grid()
#
## Secondary plot (EI Acquisition Function)
#ax2.scatter(new_points, Acquisition_EI_array, color='m', marker='o', label="EI Acquisition Function")
#ax2.set_xlabel("X_test (New Inputs)")
#ax2.set_ylabel("EI Value")
#ax2.legend(loc="upper left")
#ax2.grid()
#
#plt.tight_layout()
#plt.show()
#


#new_points = np.linspace(x_min, x_max, 10000).reshape(-1, 1)
#Acquisition_LB_deriv_list=[]
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Acquisition_LB_deriv = d_LB_acquisition(1.0,x_star_reshaped)
#
#    # Append the result to the list
#    Acquisition_LB_deriv_list.append(Acquisition_LB_deriv)
#
## Convert the list to a numpy array for convenience (optional)
#Acquisition_LB_deriv_array = np.array(Acquisition_LB_deriv_list)
#
#
#Mean_GP_list=[]
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Mean_GP = gp_surrogate_model(x_star_reshaped)
#
#    # Append the result to the list
#    Mean_GP_list.append(Mean_GP)
#
## Convert the list to a numpy array for convenience (optional)
#Mean_GP_array = np.array(Mean_GP_list)
#
#
#Mean_GP_deriv_list=[]
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Mean_GP_deriv = gp_surrogate_derivative2(x_star_reshaped)
#
#    # Append the result to the list
#    Mean_GP_deriv_list.append(Mean_GP_deriv)
#
## Convert the list to a numpy array for convenience (optional)
#Mean_GP_deriv_array = np.array(Mean_GP_deriv_list)
#
#Std_GP_list=[]
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Std_GP = gp_surrogate_covariance(x_star_reshaped)
#
#    # Append the result to the list
#    Std_GP_list.append(Std_GP)
#
## Convert the list to a numpy array for convenience (optional)
#Std_GP_array = np.array(Std_GP_list)
#
#
#Std_GP_deriv_list=[]
#for x_star in new_points:
#    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
#    x_star_reshaped = x_star.reshape(1, 1)
#
#    # Calculate the derivative for this specific point
#    Std_GP_deriv = gp_surrogate_covariance_deriv2(x_star_reshaped)
#
#    # Append the result to the list
#    Std_GP_deriv_list.append(Std_GP_deriv)
#
## Convert the list to a numpy array for convenience (optional)
#Std_GP_deriv_array = np.array(Std_GP_deriv_list)
#
#new_points = np.linspace(x_min, x_max, 50000).reshape(-1, 1)
X_old = np.sort(X_old)  # Just to ensure the values are in order

# Generate 1000 points between each consecutive pair
new_points = []

for i in range(len(X_old) - 1):
    x_start = X_old[i]
    x_end = X_old[i + 1]
    # Avoid duplicating the end of one interval as the start of the next
    points = np.linspace(x_start, x_end, 1001)[:-1]  
    new_points.append(points)

# Append the final point of the last interval
new_points.append([X_old[-1]])

# Concatenate all segments and reshape
new_points = np.concatenate(new_points).reshape(-1, 1)


P_list=[]
for x_star in new_points:
    # Reshape x_star to make sure it's 2D for the function (it should be (1, 1))
    x_star_reshaped = x_star.reshape(1, 1)

    # Calculate the derivative for this specific point
    Std_GP_deriv = np.exp(-(0.5)*gp_surrogate_derivative(x_star_reshaped)**2/gp_surrogate_covariance_deriv(x_star_reshaped))/np.sqrt(2.0*np.pi*gp_surrogate_covariance_deriv(x_star_reshaped))
#    Std_GP_deriv = np.log(Std_GP_deriv+0.00000001)
    # Append the result to the list
    P_list.append(Std_GP_deriv)

# Convert the list to a numpy array for convenience (optional)
P_array = np.array(P_list)
#P_array = (P_array - np.min(P_array)) / (np.max(P_array) - np.min(P_array))
P_array_normalized = (P_array) / np.max(P_array)



#plt.figure(figsize=(8, 5))
#plt.title("PROBHABILTTY")
#plt.plot(new_points.flatten(), P_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), P_array.flatten(), label="GP Surrogate Model", color='b')
#
#plt.show()
#


#plt.figure(figsize=(8, 5))
#plt.title("Mean and derivative")
#plt.plot(new_points.flatten(), Mean_GP_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Mean_GP_array.flatten(), label="GP Surrogate Model", color='b')
#
#
#plt.figure(figsize=(8, 5))
#plt.title("Mean and derivative")
#plt.plot(new_points.flatten(), Mean_GP_deriv_array.flatten(), label="GP Surrogate Model", color='r')
#plt.scatter(new_points.flatten(), Mean_GP_deriv_array.flatten(), color='r')
#plt.show()
#
#
#plt.figure(figsize=(8, 5))
#plt.title("Std and derivative")
#plt.plot(new_points.flatten(), Std_GP_array.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Std_GP_array.flatten(), label="GP Surrogate Model", color='b')
#
#plt.figure(figsize=(8, 5))
#plt.title("Std and derivative")
#plt.plot(new_points.flatten(), Std_GP_deriv_array.flatten(), label="GP Surrogate Model", color='r')
#plt.scatter(new_points.flatten(), Std_GP_deriv_array.flatten(), color='r')
#
#plt.show()
#




Y_list = np.array([gp_surrogate_derivative(x.reshape(1, 1)) for x in new_points]).flatten()

Y_list_cov = np.array([gp_surrogate_covariance_deriv(x.reshape(1, 1)) for x in new_points]).flatten()

std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev

# Plot the surrogate model predictions
#plt.figure(figsize=(8, 5))
#plt.plot(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.scatter(new_points.flatten(), Y_list.flatten(), label="GP Surrogate Model", color='b')
#plt.fill_between(new_points.flatten(), lower_bound, upper_bound, color='b', alpha=0.2, label="95% Confidence Interval")
#
## Labels and title for main plot
#
## Secondary plot (EI Acquisition Function)
#
#plt.show()
##
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

lbfgs_options = {
    "gtol": 1e-7,   # Stricter gradient tolerance (default is 1e-5)
    "ftol": 1e-10,  # Stricter function tolerance (default is ~2.2e-9)
    "maxiter": 20000  # Increase max iterations if needed
}


#P_array = np.array(P_list).flatten()  # Ensure it's a 1D array
#
## Get corresponding x values
#filtered_x = np.array(new_points)[P_array > -10]  # Filter x values where P > -10
#
#if len(filtered_x) > 0:  # Ensure there are valid points
#    new_x_min, new_x_max = np.min(filtered_x), np.max(filtered_x)
#    new_bounds = [(new_x_min, new_x_max)]
#else:
#    new_bounds = bounds  # If no valid points, keep the original bounds
#
#print(f"Updated bounds based on P_array filtering: {new_bounds}")
#


P_array_normalized = np.array(P_array_normalized).flatten()  # Ensure it's a 1D array
x_values = np.array(new_points).flatten()  # Ensure new_points is also 1D

# Identify regions where P_array > -10
valid_mask = P_array_normalized > 0.1  # Boolean mask of valid values
valid_x = x_values[valid_mask]  # Extract corresponding x values

# Find separate bounded intervals
if len(valid_x) > 0:
    intervals = []
    start = valid_x[0]  # Start of first interval

    for i in range(1, len(valid_x)):
        if valid_x[i] > valid_x[i - 1] + 1e-4:  # Detect a gap (adjust threshold if needed)
            intervals.append((start, valid_x[i - 1]))  # Save previous interval
            start = valid_x[i]  # Start new interval

    # Append the last interval
    intervals.append((start, valid_x[-1]))

    new_bounds = intervals
else:
    new_bounds = bounds  # If no valid intervals, keep original bounds

print(f"Updated bounded intervals based on P_array filtering: {new_bounds}")


def multi_start_minimize_LB(n_starts):
    """Find the global minimum of the LB acquisition function using Multi-Start L-BFGS-B across all intervals."""
    best_x = None
    best_fun = float("inf")


#if not new_bounds:  # Ensure new_bounds is not empty
#        new_bounds = bounds  # Fall back to original bounds

    num_intervals = len(new_bounds)
    starts_per_interval = n_starts // num_intervals  # Evenly distribute starts
    extra_starts = n_starts % num_intervals  # Remaining starts

    for i, interval in enumerate(new_bounds):
        # Assign extra start points to some intervals if needed
        num_starts = max(1, starts_per_interval + (1 if i < extra_starts else 0))
        
        for _ in range(num_starts):
            x0 = np.random.uniform(interval[0], interval[1], size=(1,))  # Initialize within interval

            result = minimize(
                lambda x: LB_acquisition(kappa, np.atleast_2d(x)),
                x0,
                bounds=[interval],  # Use the current interval
                method="L-BFGS-B",
                options=lbfgs_options
            )

            if result.fun < best_fun:  # Track best result
                best_x, best_fun = result.x, result.fun

    return best_x, best_fun


#def multi_start_minimize_LB(n_starts):
#    """Find the global minimum of the LB acquisition function using Multi-Start L-BFGS-B."""
#    best_x = None
#    best_fun = float("inf")  # Initialize with a very high value
#    
#    for _ in range(n_starts):
#        x0 = np.random.uniform(x_min, x_max, size=(1,))  # Random initial point
#        result = minimize(
#            lambda x: LB_acquisition(kappa, np.atleast_2d(x)),
#            x0,
#            bounds=bounds,
#            method="L-BFGS-B",
#            options=lbfgs_options
#        )
#        if result.fun < best_fun:  # Keep track of the best result
#            best_x, best_fun = result.x, result.fun
#    
#    return best_x, best_fun
#

# Run Multi-Start L-BFGS-B
x_min_LB, min_LB_value = multi_start_minimize_LB(n_starts=100)
print(f"Global minimum of LB acquisition at x = {x_min_LB}, value = {min_LB_value}")
data = np.column_stack((x_min_LB, gp_surrogate_model(x_min_LB), np.max(P_array)))  # Combine into a 2D array
np.savetxt("Optimal_LB", data, fmt="%.13f")



