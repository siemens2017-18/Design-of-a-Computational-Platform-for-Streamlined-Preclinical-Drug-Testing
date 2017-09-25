%%This script is for generating distributional patterns using the MYC-E2F network

% Initialize environment
clear; clc; close all;
rng('default');

% Setup constants
Sfinal=1;              % Final serum concentration
dt=0.001;              % Time step
endTime=50;            % Total time in hours
Tspan=0:dt:endTime;    % Time span
Trials=100;            % Number of iterations
mLee_E2FDist_Setup     % Script to iniitialize parameters
sigma=sqrt(1);         % Scaling for intrinsic noise
delta=sqrt(5);         % Scaling for extrinsic noise

% Main program
kMYC=zeros(1,Trials);
for i=1:Trials
    kMYC(i)=randn;
end
data=zeros(Trials,2);
for i=1:Trials
    rMYC = Z*10^kMYC(i);
    x = mLee_E2FDist_kMYCStochSim(dt, Tspan, x0, Sfinal, rMYC, ...
        paraset, sigma, delta);
    data(i,1) = x(end,1);
    data(i,2) = x(end,2);
end
csvwrite('LeeMYC-E2FData.csv',data);