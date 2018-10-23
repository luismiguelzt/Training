function [m_n,P_n] = kalman_filter (m_n1,P_n1,y_n,F_n,Q_n,H_n,R_n)

% Kalman filter prediction and update steps --- propagates Gaussian
% posterior state distribution from time step n-1 to time step n
%
% Inputs:
% m_n1 - (Dx1 vector) Gaussian posterior mean at time step n-1
% P_n1 - (Dx1 vector) Gaussian posterior covariance at time step n-1
% y_n - (Mx1 vector) Measurements at time step n
% F_n - (DxD matrix) state-transition matrix at time step n
% Q_n - (DxD matrix) Gaussian state noise covariance at time step n
% H_n - (MxD matrix) Measurement matrix at time step n
% R_n - (MxM matrix) Gaussian measurement noise covariance at time step n
%
% Outputs:
% m_n - (Dx1 vector) Gaussian posterior mean at time step n
% P_n - (Dx1 vector) Gaussian posterior covariance at time step n

% Predict
m_nn1 = F_n*m_n1;
P_nn1 = F_n*P_n1*F_n' + Q_n;

% Update
S_n = H_n*P_nn1*H_n' + R_n;
K_n = P_nn1*H_n'/S_n;
m_n = m_nn1 + K_n*(y_n-H_n*m_nn1);
P_n = P_nn1 - K_n*H_n*P_nn1;