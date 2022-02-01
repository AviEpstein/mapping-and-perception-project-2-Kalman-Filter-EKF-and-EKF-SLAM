# mapping-and-perception-project-2-Kalman-Filter-EKF-and-EKF-SLAM
Abstract
In this project we will be implementing and analyzing the following Algorithmes:
● Kalman Filter
● Extended kalman filter
● EKF - SLAM
In the first section we will implement the classic kalman filter, we will extract the KITTI OXTS 
GPS trajectory from recorded data 2011_09_26_drive_0022, from this data we will get the
- lat: latitude of the oxts-unit (deg)
- long: longitude of the oxts-unit (deg) 
- Extract timestamps from KITTI data and convert them to seconds elapsed from the first 
one
We will then transform these LLA coordinates to ENU coordinate system, add Gaussian noise to 
x and y of the ENU coordinates and then implement the kalman filter on the constant velocity 
model given the noise trajectories in order to approximate the ground truth trajectory we will see 
how to calibrate and initialize the appropriate matracis and initial conditions in order to minimize 
the RMSE error as best as possible to get a maxE less than 7. We will also see the result of the 
covariance matrix of state vector and dead reckoning the kalman gain after 5 sec and how it 
impacts the estimated trajectory, and analyze the estimated x-y values separately and 
corresponding sigma value along the trajectory.
As a bonus i will implement the same problem for the constant acceleration model.
In the second section we will implement the Extended kalman filter and as the same as in the 
first section we will use the noised trajectories as inputs of the same data as in section 1 we will 
see how we can deal with a nonlinear motion model and still apply kalman filter on it (EKF) 
again we will initialize and compute the appropriate matrices and plot the same results and 
analysis as in section 1 and see if we can reduce the RMSE and maxE to get a better 
approximation and how it deals differently with dead reckoning.
In the third section we will implement the EKF - SLAM algorithm, we will run the odometry 
motion model where the inputs “u” are gaussian noised, and we have assumed measurements 
from our state with some Gaussian noise. Here will implement the predicted state of our motion 
and then implement the correction of our state computing the effect of each observed landmark 
on the kalman gain, corrected mean and uncertainty matrix, for each time step to fully compute 
the estimated localization and mapping. Our inputs will be observations at the relevant time 
steps and motion commands. We will then analyze the results to reach minimum RMSE and 
maxE values. Moreover will analyze the estimation error of X,Y,Theta and of 2 landmarks



# for more information about result analysis algorithim and code read the full project 2 pdf report
