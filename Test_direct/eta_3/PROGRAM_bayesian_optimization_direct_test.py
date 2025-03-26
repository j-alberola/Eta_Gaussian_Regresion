import numpy as np
#import matplotlib.pyplot as plt
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

#plt.show()
#Optimization of the hyperparameters + fixing the noise to 0
m = GPy.models.GPRegression(X,velocity,kernel)
m.Gaussian_noise.variance.fix(0.0)

#m.likelihood.variance = 1e-1
from IPython.display import display

#m.optimize(messages=True)

m.optimize_restarts(messages=False,verbose=False, max_iters=1000, optimizer='bfgs', num_restarts=40)

#display(m)
#fig = m.plot()
#print("Noise variance:", m.Gaussian_noise.variance.values)
#plt.show()
#
##Defining the number of points used in the plots of the Surrogate Model
x_min = np.min(X)
x_max = np.max(X)
new_points = np.linspace(x_min, x_max, 1000).reshape(-1, 1)


K = kernel.K(X,X)
K_imag = kernel_imag.K(X,X)


from scipy.linalg import cho_solve, cholesky

# Compute Cholesky decomposition
L = cholesky(K + np.eye(len(X)) * 1e-6, lower=True)  # Add jitter for stability

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

    L = cholesky(K + np.eye(len(X)) * 1e-10, lower=True)
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

Y_list = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in new_points]).flatten()
Y_list_training = np.array([gp_surrogate_model(x.reshape(1, 1)) for x in X_old]).flatten()

Y_list_cov = np.array([gp_surrogate_covariance(x.reshape(1, 1)) for x in new_points]).flatten()

std_dev = np.sqrt(Y_list_cov)
upper_bound = Y_list + 1.96 * std_dev
lower_bound = Y_list - 1.96 * std_dev

# Ensure Y_list is in correct shape
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

    L = cholesky(K + np.eye(len(X)) * 1e-8, lower=True)

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

#Computing the velocity with the corresponding uncertanty


def gp_surrogate_velocity_negative(x):
    # Normalize input
    x = np.atleast_2d(x).T
    velocity = -gp_surrogate_model(x)
    return velocity

#from matplotlib import gridspec

#
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def EI_acquisition(x, Min_Y,xi=0.0):
    """Compute the Expected Improvement (EI) acquisition function."""
    mean = gp_surrogate_velocity_negative(x)  # Assuming negative velocity is being maximized
    std = np.sqrt(gp_surrogate_covariance(x))  # Use the known variance expression
    

    if std < 1e-11:  # Avoid division by zero
        return max(mean - np.max(Max_Y) - xi, 0)

    f_best = np.max(Min_Y)  # Best observed function value (minimization)
    
    improvement = mean - f_best - xi  # Difference from best observed
    Z = improvement / std  # Standardized improvement
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)  # EI formula
    
    return max(ei, 0)  # Ensure EI is non-negative


def LB_acquisition(kappa, x):
    """Compute the UCB acquisition function using the known function and variance."""
    mean = gp_surrogate_model(x)  # Assuming negative velocity is being maximized
    std = np.sqrt(gp_surrogate_covariance(x))  # Use the known variance expression
    return mean - kappa * std  # UCB formula


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
    Acquisition_EI = EI_acquisition(x_star_reshaped, -Velocity_array_old)

    # Append the result to the list
    Acquisition_EI_list.append(Acquisition_EI)

# Convert the list to a numpy array for convenience (optional)
Acquisition_EI_array = np.array(Acquisition_EI_list)

#print(Acquisition_EI_array)

## Calculating and plotting 1st and 2nd derivative of the velocity 
#
# Computing the minimas of the surrogate model (between the first and last training points)
#


bounds = [(x_min, x_max)]
kappa = 3.0
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def multi_start_minimize_LB(n_starts=10):
    """Find the global minimum of the LB acquisition function using Multi-Start L-BFGS-B."""
    best_x = None
    best_fun = float("inf")  # Initialize with a very high value
    
    for _ in range(n_starts):
        x0 = np.random.uniform(x_min, x_max, size=(1,))  # Random initial point
        result = minimize(
            lambda x: LB_acquisition(kappa, np.atleast_2d(x)),
            x0,
            bounds=bounds,
            method="L-BFGS-B"
        )
        if result.fun < best_fun:  # Keep track of the best result
            best_x, best_fun = result.x, result.fun
    
    return best_x, best_fun

def multi_start_maximize_EI(n_starts=10):
    """Find the global maximum of the EI acquisition function using Multi-Start L-BFGS-B."""
    best_x = None
    best_fun = -float("inf")  # Initialize with a very low value
    
    for _ in range(n_starts):
        x0 = np.random.uniform(x_min, x_max, size=(1,))  # Random initial point
        result = minimize(
            lambda x: -EI_acquisition(np.atleast_2d(x), -Velocity_array_old),  # Negate EI for maximization
            x0,
            bounds=bounds,
            method="L-BFGS-B"
        )
        if -result.fun > best_fun:  # Keep track of the best result
            best_x, best_fun = result.x, -result.fun  # Revert negation
    
    return best_x, best_fun

# Run Multi-Start L-BFGS-B
x_min_LB, min_LB_value = multi_start_minimize_LB(n_starts=10)
print(f"Global minimum of LB acquisition at x = {x_min_LB}, value = {min_LB_value}")

x_max_EI, max_EI_value = multi_start_maximize_EI(n_starts=10)
print(f"Global maximum of EI acquisition at x = {x_max_EI}, value = {max_EI_value}")
#
np.savetxt("Optimal_LB", x_min_LB, fmt="%.10f")
np.savetxt("Optimal_EI", x_max_EI, fmt="%.10f")

