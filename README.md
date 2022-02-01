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
maxE values. Moreover will analyze the estimation error of X,Y,Theta and of 2 landmarks.
Contents
תוכן עניינים
Abstract.........................................................................................................................................................2
Contents........................................................................................................................................................3
List of Figures...............................................................................................................................................4
List of Movies...............................................................................................................................................4
List of Tables................................................................................................................................................4
:Solutions......................................................................................................................................................5
: Kalman Filter..............................................................................................................................................5
: Etended Kalman Filter..............................................................................................................................12
: EKF - SLAM ............................................................................................................................................22
Summary.....................................................................................................................................................31
Code ............................................................................................................................................................32
Appendix.....................................................................................................................................................56
List of Figures
Figure 1: World coordinate (LLA) 
Figure 2: Local coordinate (ENU) 
Figure 3: graph of original GT and observations noise
Figure 4: Ground-truth and estimated results
Figure 5: xestimated-xGT and 𝜎𝑥 values
Figure 6: yestimated-yGT and 𝜎y values
Figure 7: world coordinate (LLA)
Figure 8: Local coordinate (ENU)
Figure 9: ground-truth yaw angles
Figure 10: ground-truth forward velocities
Figure 11: ground-truth yaw rates
Figure 12: EKF results no noise in commands
Figure 13: graphs of GT+ noise yaw rate
Figure 14: graphs of GT+ noise forward velocities
Figure 15: EKF results with noise in commands
Figure 16: xestimated-xGT and 𝜎𝑥 values
Figure 17: yestimated-yGT and 𝜎y values
Figure 18: thetaestimated-thetaGT and 𝜎theta values
Figure 19: odomatry motion model GT
Figure 20: estimation error of x
Figure 21: estimation error of y
Figure 22: estimation error of theta
Figure 23: estimation error of landmark 1 x
Figure 24: estimation error of landmark 1 y
Figure 25: estimation error of landmark 2 x
Figure 26: estimation error of landmark 2 y
List of Movies
Attached movies:
- animation of GT KF estimate and dead reckoning
- animation of GT EKF estimate and dead reckoning
- animation of Trajectory of EKF-SLAM
List of Tables
Table 1: RMSE, maxE vs sigma_n
Solutions:
Kalman Filter :
a) Recorded data: 2011_09_26_drive_0022 KITTI GPS sequence (OXT) was downloaded.
b) In this section we have extracted vehicle GPS trajectory from KITTI OXTS senser 
packets which are treated as ground truth in this experiment:
- lat: latitude of the oxts-unit (deg) 
- long: longitude of the oxts-unit (deg) 
- Extract timestamps from KITTI data and convert them to seconds elapsed from 
the first one
c) Here we transformed the GPS trajectory from [lat, long, alt] to local [x, y, z] ENU 
coordinates in order to enable the Kalman filter to handle them and plotted the GT LLA 
and ENU coordinates.
Figure 1: World coordinate (LLA) 
Figure 2: Local coordinate (ENU) 
We can see that the vehicle drove North 20m -> West for 175m -> NNE 60m and -> East 
180m -> NNE 60m -> WWN 30m .
d) Here we added Gaussian noise to the ground-truth GPS data which will be used as 
noisy observations fed to the Kalman filter. Noise added with standard deviation of 
observation noise of x and y in meter (𝜎𝑥 = 3 , 𝜎𝑦 = 3). In the next figure we can see 
the original GT and observations noise:
Figure 3: graph of original GT and observations noise
e) Here we will apply a linear Kalman filter to the GPS sequence in order to estimate 
vehicle 2D pose based on constant velocity model
Will suppose initial 2D position [x, y] estimation starts with the first GPS observation (the 
noised one), GPS observation noise of X and Y is known (𝜎𝑥 =3, 𝜎𝑦 =3).
Our goal will be to minimize the RMSE which is defined as:
1) Initial conditions: according to your first observation the values of standard 
deviations initialized:
𝜇̅= [
𝑥0
𝑣𝑥0
𝑦0
𝑣𝑦0
] = [
𝑥𝑒𝑠𝑡0
0
𝑦𝑒𝑠𝑡0
1
]
This is because we can see from the initial observation that at the beginning of 
the drive is north so we set 𝑣𝑥0 to zero and assumed a value for 𝑣𝑦0 to 1.
Σ0 = [
𝜎𝑥
2 0 0 0
0 100 0 0
0 0 𝜎𝑦
2 0
0 0 0 100
]
This is because we want our first covariance uncertainty's of our state to be 
relatively hi at the beginning as we have not yet got corrections of our state so 
we are less certain of our initial condition after we run the algorithm this 
uncertainty covariance's will converge to contain 66% of the error if it contains 
more than 66 percent we can decrees the appropriate uncertainty. hence we 
chose the uncertainty of X and Y according to the variance of the measurements 
and for the velocities we choose a hi enough number to be able to handle the 
unknown velocities.
2) Matrixes: A ,B ,C:
𝐴 = [
1 ∆𝑡 0 0
0 1 0 0
0 0 1 ∆𝑡
0 0 0 1
] , 𝐵 = 𝑁𝑜𝑛𝑒 (𝑐𝑜𝑛𝑠𝑡 𝑣𝑒𝑙𝑜𝑐𝑖𝑡𝑦) , 𝐶 = [
1 0 0 0
0 0 1 0
]
Corresponding to const velocity model 𝜇̅𝑡 = 𝐴𝑡𝜇𝑡−1 + 𝐵𝑡𝑢𝑡 while we only observe 
x and y form here matrix C.
3) Measurement covariance (Q):
𝑄 = [
𝜎𝑥
2 0
0 𝜎𝑦
2
]
Corresponding to the measurement noise, we only measure x and y hence 
matrix of size 2x2.
4) transition noise covariance R:
containing the process noise in the const velocity model this is the source of the 
change in speed making it dynamic hence after analyzing different values 
𝑅 = [
0 0 0 0
0 ∆𝑡 0 0
0 0 0 0
0 0 0 ∆𝑡
] ∗ 𝜎𝑛
2
, 𝜎𝑛
2 = 1
You can see in the next graphs the values RMSE an maxE compared to values 
of 𝜎𝑛:
Table 1: RMSE, maxE vs sigma_n
Hence sigma_n was set to 1
Kalman filter main routine:
the state mean and uncertainty covariance matrix was initialized as above sections, from here 
the function the performe_KalmanFilter was called this function organized the inputs of the 
initial state and created the list of states and covariance's and then iterated over all time steps 
and ran the one step one_step_of_KalmanFilter and saved the state and covariance's.
the kalman step itself contains 
the prediction of the location and uncertainty covariance by calculating
