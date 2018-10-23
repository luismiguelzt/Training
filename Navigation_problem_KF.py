import numpy as np
import matplotlib.pyplot as plt

#from kalman_filter import kalman_filter
def kalman_filter(m_n1,P_n1,y_n,F_n,Q_n,H_n,R_n):
    # Predict
    
    m_nn1 = F_n[:,:,0].dot(m_n1)
    P_nn1 = (F_n[:,:,0].dot(P_n1[:,:,0])).dot(F_n[:,:,0].T) + Q_n
    
    # Update
    
    S_n = (H_n.dot(P_nn1)).dot(H_n.T) + R_n
    K_n = (P_nn1.dot(H_n.T)).dot(np.linalg.inv(S_n))
    m_n = m_nn1 + (K_n.dot((y_n-(H_n.dot(m_nn1))).T)).T
    P_n = P_nn1 - (K_n.dot(H_n)).dot(P_nn1)
    mP_n = np.c_[m_n.T,P_n]
    #mP_n = mP_n[:,:,np.newaxis]
        
    return mP_n

# Radar tracking using Kalman Filter

# State Space Model

delt = 0.5                                                                  #Time step interval
F_n = np.array([[1.,delt],[0,1.]])                                          #State-transition matrix
F_n = F_n[:,:,np.newaxis]
sigma_x = 0.1
Q_n = sigma_x*np.array([[delt**2/3,delt**2/2],[delt**2/2,delt]])            #State noise covariance matrix
H_n = np.identity(2)                                                        #Measurement matrix
sigma_y = 0.1
R_n = sigma_y**2*np.identity(2)

# Initialization

x = np.array([[1],[1]])
m = x[0:2,0]
P = np.identity(len(x))
P = P[:,:,np.newaxis]

#Generate an empty array to store the results of the mean and covariance
R = np.empty((2,3,1))

#Track target using Kalman Filter

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

# Plot actual and estimated target position and velocity
line1 = plt.plot(x[0,],x[1,],'r-', label='Actual')
line2 = plt.plot(R[0,0,],R[1,0,],'b--', label='Estimated')
plt.xlabel('Position X (m)')
plt.ylabel('Velocity V (m/s)')
plt.title('Actual and Estimated Target Position and Velocity')
plt.legend()
plt.show()