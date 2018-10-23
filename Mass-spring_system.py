#Mass-Spring System Solved Using Linear Kalman Filtering.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system parameters

ma = 10
k = 5
b = 3
A = np.array([[0,1],[-k/ma,-b/ma]])

# Simulate system.

rhs = lambda t,x: A.dot(x)    #Right hand side of the partial differential equation

# Find the states using analytical model

xinit = np.array([1,0])         # Vector [NX1] - Initial position and velocity
T = 30                          # Final time
delt = 0.2                         # Time step
time = np.arange(0,T,delt)         # Vector [MX1] - Initial time

sol = solve_ivp(rhs, [0,T], xinit, t_eval=time)    # Stores the values of the times (sol.t) and corresponding solution values at time T (sol.y).

# Generating model and measurement data

sigma_x = 0.1                  # Standard deviation of the noise in the model.
x = sol.y[:,0]                 # Initial observation of the system which coincides with the initial state.
F_n = (np.identity(len(x))+delt*A) # State transformation matrix
Q_n = sigma_x*np.identity(len(x))  # State noise covariance matrix
m = x[0:2,0]                   # Mean of the process
P = np.identity(len(x))
P = P[:,:,np.newaxis]

#Generate an empty array to store the results of the mean and covariance
R = np.empty((2,3,1))

#Track displacement using Kalman Filter

for n in range(1,100):
#    print n
    
    # State Propagation
    v_n = np.random.multivariate_normal([0,0], Q_n)
    x_n = F_n[:,:,0].dot(x[0:2,n-1]) + v_n
    x = np.c_[x,x_n]                                                    # Actual position and velocity

    # Generate measurements
    w_n = sigma_y*np.random.randn(2,1)
    y_n = H_n.dot(x_n) + w_n.T
    
    #Compute Gaussian posterior mean and covariance at time step n
    if n==1:
        R[:,:,n-1] = kalman_filter(m,P,y_n,F_n,Q_n,H_n,R_n)
    else:
        dummy = kalman_filter(R[0:2,0,n-2],np.reshape(R[0:2,1:3,n-2],(2L,2L,1L)),y_n,F_n,Q_n,H_n,R_n)
        R = np.dstack((R,dummy))
    
#print R






