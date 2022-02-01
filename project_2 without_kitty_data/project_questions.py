import os
from data_preparation import *
from kalman_filter import *
import graphs
#import numpy as np


class ProjectQuestions:
    def __init__(self, dataset):
        self.dataset = dataset
        # Extract vehicle GPS trajectory from KITTI OXTS senser packets.
        self.LLA_GPS_trajectory = build_LLA_GPS_trajectory(self.dataset)

        # Transform GPS trajectory from [lat, long, alt] to local [x, y, z] coordinates in order to enable the Kalman filter to handle them.
        self.ENU_locations_array, self.times_array , self.yaw_vf_wz_array  = build_GPS_trajectory(self.dataset)
        self.number_of_frames = self.ENU_locations_array.shape[0]
    
    def Q1(self ,basedir):
        """
        That function runs the code of question 1 of the project.
        Loads from kitti dataset, set noise to GT-gps values, and use Kalman Filter over the noised values.
        """

        # Plot ground-truth GPS trajectory
        graphs.plot_single_graph(self.LLA_GPS_trajectory,'World coordinate (LLA)','lon','lat','World coordinate (LLA)')
        graphs.plot_single_graph(self.ENU_locations_array,'Local coordinate (ENU)','X [m]','Y [m]','Local coordinate (ENU)')

        # Add gaussian noise to the ground-truth GPS data:
        sigma_x_y = [3,3]
         
        self.ENU_locations_array_noised = np.concatenate((add_gaussian_noise(self.ENU_locations_array[:,0],sigma_x_y[0]).reshape(self.number_of_frames,1),add_gaussian_noise(self.ENU_locations_array[:,1],sigma_x_y[1]).reshape(self.number_of_frames,1)),axis = 1)
        
        
        # plot on the same graph original GT and observations noise.
        graphs.plot_graph_and_scatter(self.ENU_locations_array,self.ENU_locations_array_noised,'original GT and observations noise','X [m]','Y [m]','ENU_GT', 'ENU_noised',)

        #print(self.ENU_locations_array_noised.shape)
        
        # sigma_x_y, sigma_Qn = 
        
        predicted_mean_0 = np.array([self.ENU_locations_array_noised[0][0],0,self.ENU_locations_array_noised[0][1],1])
        predicted_uncertinty_0 = np.array([ [math.pow(sigma_x_y[0],2),0,0,0], [0,100,0,0], [0,0,math.pow(sigma_x_y[1],2),0],[0,0,0,100] ])
        
        #observation_z_t_array = np.concatenate((self.ENU_locations_array_noised[:,0].reshape(self.number_of_frames,1),np.zeros((self.ENU_locations_array_noised.shape[0],1)),self.ENU_locations_array_noised[:,1].reshape(self.number_of_frames,1),np.zeros((self.ENU_locations_array_noised.shape[0],1))),axis = 1)
        observation_z_t_array = self.ENU_locations_array_noised   
        #print(observation_z_t_array)
        #print(observation_z_t_array.shape)
        KF = KalmanFilter()
        
        # KalmanFilter 
       
        maxE_list = []
        RMSE_list = []
    
        for i in range(1,20,1):
          sigma_n = i
          X_Y_est , uncertinty_cov_list = KalmanFilter.performe_KalmanFilter(KF,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n) 
          X_Y_est = X_Y_est[:,[0,2]]
          #print("uncertinty_cov_list.shape" ,uncertinty_cov_list.shape)
          #print("uncertinty_cov_list: ", uncertinty_cov_list)     
          RMSE, maxE = KalmanFilter.calc_RMSE_maxE(self.ENU_locations_array,X_Y_est)
          maxE_list.append(maxE)
          RMSE_list.append(RMSE)

        maxE_array = np.array(maxE_list, ndmin = 2).T
        RMSE_array = np.array(RMSE_list, ndmin = 2).T
        sigma_n_array = np.array([i for i in range(1,20,1)], ndmin = 2).T
        maxE_array_sigma_n = np.concatenate((sigma_n_array,maxE_array),axis=1)
        RMSE_array_sigma_n = np.concatenate((sigma_n_array, RMSE_array),axis=1)
        graphs.plot_single_graph(maxE_array_sigma_n, 'maxE vs sigma_n', 'sigma_n','maxE', 'maxE vs sigma_n')
        graphs.plot_single_graph(RMSE_array_sigma_n,'RMSE vs sigma_n','sigma_n','RMSE', 'RMSE vs sigma_n')
        print("maxE_array shape", maxE_array.shape)
        
        sigma_n = np.argmin(maxE_list)+1
        
        X_Y_est , uncertinty_cov_list = KalmanFilter.performe_KalmanFilter(KF,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n) 
        X_Y_est = X_Y_est[:,[0,2]]
        print("RMSE " ,RMSE_list[np.argmin(maxE_list)] ,"maxE" , min(maxE_list), 'sigma_n' , sigma_n)

        

        # build_ENU_from_GPS_trajectory
        graphs.plot_three_graphs( self.ENU_locations_array, X_Y_est,self.ENU_locations_array_noised, 'KF esults:' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        
        # make dead reckoning after 5 seconds (inserting dead reckoning = true):
        KF_dead_reckoning = KalmanFilter()
        X_Y_est_dead_reckoning , uncertinty_cov_list_dead_reckoning = KalmanFilter.performe_KalmanFilter(KF_dead_reckoning,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n, dead_reckoning = True )
        X_Y_est_dead_reckoning = X_Y_est_dead_reckoning[:,[0,2]]
        
        # make animation from gt est and est dead reckoning:
        X_Y_GT_locations = self.ENU_locations_array[:,:2]
        print("X_Y_GT_locations[0] " , X_Y_GT_locations[0], "X_Y_est[0] ", X_Y_est[0],"X_Y_est_dead_reckoning", X_Y_est_dead_reckoning[0], "uncertinty_cov_list[0] " ,uncertinty_cov_list[0] )
        X_XY_XY_Y_uncertinty_cov_list = uncertinty_cov_list[:,[0,2,8,10]]
        ani = graphs.build_animation(X_Y_GT_locations, X_Y_est, X_Y_est_dead_reckoning, X_XY_XY_Y_uncertinty_cov_list, 'trajectories', 'X [m]', 'Y [m]', 'GT', 'KF_estimat', 'dead reckoning')
        graphs.save_animation(ani, basedir, 'animation of GT KF estimate and dead reckoning')
        
        # Plot and analyze the estimated x-y values separately and corresponded
        # sigma value along the trajectory. (e.g. show in same graph xestimated-xGT and
        # 𝜎𝑥 values and explain your results):
        X_Y_estimated_minus_X_Y_GT = X_Y_est - self.ENU_locations_array[:,:2]
        times_array = self.times_array.reshape((self.times_array.shape[0],1))

        X_estimate_minus_X_GT = X_Y_estimated_minus_X_Y_GT[:,0].reshape((X_Y_estimated_minus_X_Y_GT[:,0].shape[0],1))
        X_estimate_minus_X_GT_and_times = np.concatenate((times_array,X_estimate_minus_X_GT),axis =1)
        sigma_x = X_XY_XY_Y_uncertinty_cov_list[:,0]
        sigma_x = np.reshape(sigma_x,(sigma_x.shape[0],1))
        sigma_minus_x = (-sigma_x)
        sigma_x_with_times = np.concatenate((times_array, sigma_x),axis =1)
        sigma_minus_x_with_times = np.concatenate((times_array, sigma_minus_x),axis =1)
        graphs.plot_two_graphs_one_double(X_estimate_minus_X_GT_and_times,sigma_x_with_times,sigma_minus_x_with_times,'X estimated - XGT and sigma_x values', 'Times elapsed [sec]', 'X estimation error [m]' ,'estimation eror', 'estimated 1 sigma interval')


        #Y_estimate_minus_Y_GT_and_times = hstack(self.times_array, X_Y_estimated_minus_X_Y_GT[1])
        Y_estimate_minus_Y_GT = X_Y_estimated_minus_X_Y_GT[:,1].reshape((X_Y_estimated_minus_X_Y_GT[:,1].shape[0],1))
        Y_estimate_minus_Y_GT_and_times = np.concatenate((times_array,Y_estimate_minus_Y_GT),axis =1)
        sigma_y = X_XY_XY_Y_uncertinty_cov_list[:,3]
        sigma_y = np.reshape(sigma_y,(sigma_y.shape[0],1))
        sigma_minus_y = (-sigma_y)
        sigma_y_with_times = np.concatenate((times_array, sigma_y),axis =1)
        sigma_minus_y_with_times = np.concatenate((times_array, sigma_minus_y),axis =1)
        graphs.plot_two_graphs_one_double(Y_estimate_minus_Y_GT_and_times,sigma_y_with_times,sigma_minus_y_with_times,'Y estimated - YGT and sigma_y values', 'Times elapsed [sec]', 'Y estimation error [m]' ,'estimation eror', 'estimated 1 sigma interval')
        
        # (bonus! 5%). Implement constant-acceleration model and compare the
        # results with constant-velocity model
        predicted_mean_0 = np.array([self.ENU_locations_array_noised[0][0],0,0,self.ENU_locations_array_noised[0][1],1,0])
        predicted_uncertinty_0 = np.array([ [3*math.pow(sigma_x_y[0],2),0,0,0,0,0], [0,100,0,0,0,0],[0,0,100,0,0,0] ,[0,0,0,3*math.pow(sigma_x_y[1],2),0,0],[0,0,0,0,100,0],[0,0,0,0,0,100] ])
        X_Y_est , uncertinty_cov_list = KalmanFilter.performe_KalmanFilter(KF,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n,const_acc =True) 
        X_Y_est = X_Y_est[:,[0,3]]
        RMSE, maxE = KalmanFilter.calc_RMSE_maxE(self.ENU_locations_array,X_Y_est)
        print("RMSE const acc" , RMSE, "maxE const acc" , maxE)
        


     
    def Q2(self,basedir):

      # 2.a+2.b performed in intlization
  

        #2.c Plot ground-truth GPS trajectory Plot ground-truth yaw angles, yaw rates, and forward velocities
        graphs.plot_single_graph(self.LLA_GPS_trajectory,'World coordinate (LLA)','lon','lat','World coordinate (LLA)')
        graphs.plot_single_graph(self.ENU_locations_array,'Local coordinate (ENU)','X [m]','Y [m]','Local coordinate (ENU)')
        
        times_array = self.times_array.reshape((self.times_array.shape[0],1))
        yaw_array = np.array(self.yaw_vf_wz_array[:,0],ndmin = 2).T
        vf_array = np.array(self.yaw_vf_wz_array[:,1],ndmin = 2).T
        wz_array = np.array(self.yaw_vf_wz_array[:,2],ndmin = 2).T
        
        yaw_array_and_times = np.concatenate((times_array,yaw_array),axis = 1)
        vf_array_and_times = np.concatenate((times_array,vf_array),axis = 1)
        wz_array_and_times = np.concatenate((times_array,wz_array), axis = 1)
        
        graphs.plot_single_graph(yaw_array_and_times,'ground-truth yaw angles', 'time elapsed [sec]','yaw angle [rad]','yaw angles GT' )
        graphs.plot_single_graph(vf_array_and_times,'ground-truth forward velocity ', 'time elapsed [sec]','forward velocity [m/s]','forward velocity GT' )
        graphs.plot_single_graph(wz_array_and_times,'ground-truth yaw rates', 'time elapsed [sec]','yaw rate [rad/s]','yaw rates GT' )

        # 2.d. Add gaussian noise to the ground-truth GPS/IMU data.  Those are used as noisy observations given to Kalman filter later. standard deviation of observation noise of x and y in meter
        sigma_x_y = [3,3]        
        self.ENU_locations_array_noised = np.concatenate((add_gaussian_noise(self.ENU_locations_array[:,0],sigma_x_y[0]).reshape(self.number_of_frames,1),add_gaussian_noise(self.ENU_locations_array[:,1],sigma_x_y[1]).reshape(self.number_of_frames,1)),axis = 1)
        #print(self.ENU_locations_array_noised.shape)
        #X_Y_theta_noised = np.concatenate((self.ENU_locations_array_noised,yaw_array), axis =1)
        
        # intializtion:
        predicted_mean_0 = np.array([self.ENU_locations_array_noised[0][0],self.ENU_locations_array_noised[0][1],yaw_array[0][0]])
        predicted_uncertinty_0 = np.array([ [math.pow(sigma_x_y[0],2),0,0], [0,math.pow(2,2),0], [0,0,0.0174533] ])
        
        observation_z_t_array = self.ENU_locations_array_noised   
        sigma_n = 0
        ekf = ExtendedKalmanFilter()  
        X_Y_theta_est , uncertinty_cov_list = ExtendedKalmanFilter.performe_KalmanFilter(ekf,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n, EFK = True, vf_array = vf_array, wz_array = wz_array) 
        X_Y_est = X_Y_theta_est
        #print("self.ENU_locations_array.shape" , self.ENU_locations_array.shape)
        #print("X_Y_theta_est.shape" , X_Y_theta_est.shape)
        X_Y_theta_GT = np.concatenate((self.ENU_locations_array[:,:2],yaw_array),axis= 1)
        #print("X_Y_wz_GT shape", X_Y_wz_GT.shape)
        # build_ENU_from_GPS_trajectory
        graphs.plot_three_graphs( self.ENU_locations_array, X_Y_theta_est,self.ENU_locations_array_noised, 'EKF results no noise in commands:' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        RMSE, maxE = KalmanFilter.calc_RMSE_maxE(X_Y_theta_GT,X_Y_theta_est)
        print ("EKF: no noise in command","RMSE: ", RMSE, "maxE: ",maxE )
        # need to update rmse ans maxE calc to meet all gruond truths.


        # f. Add gaussian noise to the IMU data: (5%) 
        # Add noise to yaw ratesstandard deviation of yaw rate in rad/s (𝜎𝑤 =0.2) plot graphs of GT+ noise yaw rate
        wz_array_noised = add_gaussian_noise(self.yaw_vf_wz_array[:,2],0.2).reshape(self.yaw_vf_wz_array[:,2].shape[0],1)
        #print(wz_array_noised.shape)
        wz_array_noised_and_times = np.concatenate((times_array,wz_array_noised),axis = 1)
        graphs.plot_graph_and_scatter(wz_array_and_times, wz_array_noised_and_times,'ground-truth and noised yaw rates', 'time elapsed [sec]','yaw rate [rad/s]','yaw rates GT','yaw rates noised' )

        # Add noise to forward velocitiesadd standard deviation of forward velocity in m/s (𝜎𝑓𝑣 =2) plot graphs of GT+ noise velocities
        vf_array_noised = add_gaussian_noise(self.yaw_vf_wz_array[:,1],2).reshape(self.yaw_vf_wz_array[:,1].shape[0],1)
        vf_array_noised_and_times = np.concatenate((times_array,vf_array_noised),axis = 1)
        graphs.plot_graph_and_scatter(vf_array_and_times, vf_array_noised_and_times,'ground-truth and noised forward velocity', 'time elapsed [sec]','forward velocity [m/s]','forward velocity GT','forward velocity noised' )


        X_Y_theta_est , uncertinty_cov_list = ExtendedKalmanFilter.performe_KalmanFilter(ekf,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n , EFK = True, vf_array = vf_array_noised, wz_array = wz_array_noised, sigma_vf = 2, sigma_wz = 0.2, yaw_rate_and_vf_noised = True)         
        X_Y_est = X_Y_theta_est[:,:2]
        graphs.plot_three_graphs( X_Y_theta_GT, X_Y_theta_est,self.ENU_locations_array_noised, 'EKF results with noise in commands:' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        RMSE, maxE = KalmanFilter.calc_RMSE_maxE(X_Y_theta_GT,X_Y_est)
        print ("EKF: with noise in command ","RMSE: ", RMSE, "maxE: ",maxE )
        sigma_n = 0.12
        maxE_list = []
        RMSE_list = []
        k_array = []
        for i in range(1,20,1):
          
          k = i
          #predicted_uncertinty_0 = np.array([ [k*math.pow(sigma_x_y[0],2),0,0], [0,k*math.pow(sigma_x_y[1],2),0], [0,0,0.3*k] ])
          predicted_uncertinty_0 = np.array([ [k*math.pow(sigma_x_y[0],2),0,0], [0,k*math.pow(2,2),0], [0,0,0.0174*k] ])

          X_Y_theta_est , uncertinty_cov_list = ExtendedKalmanFilter.performe_KalmanFilter(ekf,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n , EFK = True, vf_array = vf_array_noised, wz_array = wz_array_noised, sigma_vf = 2, sigma_wz = 0.2, yaw_rate_and_vf_noised = True)         
          #X_Y_theta_est = X_Y_theta_est[:,[0,3]]
          #print("uncertinty_cov_list.shape" ,uncertinty_cov_list.shape)
          #print("uncertinty_cov_list: ", uncertinty_cov_list)     
          RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(X_Y_theta_GT,X_Y_theta_est)
          
          maxE_list.append(maxE)
          RMSE_list.append(RMSE)
          k_array.append(k)
        maxE_array = np.array(maxE_list, ndmin = 2).T
        RMSE_array = np.array(RMSE_list, ndmin = 2).T
        k_array = np.array(k_array, ndmin = 2).T
        maxE_array_k = np.concatenate((k_array,maxE_array),axis=1)
        RMSE_array_k = np.concatenate((k_array, RMSE_array),axis=1)
        graphs.plot_single_graph(maxE_array_k, 'maxE vs coaficiant k', 'k','maxE', 'maxE vs k')
        graphs.plot_single_graph(RMSE_array_k,'RMSE vs coaficaiant k','k','RMSE', 'RMSE vs k')
        #get the best estimate:
        k = np.argmin(maxE_list)+1
        #print("k", k "but realy equals 1")
        k=1
        
        #predicted_uncertinty_0 = np.array([ [k*math.pow(sigma_x_y[0],2),0,0], [0,k*math.pow(sigma_x_y[1],2),0], [0,0,1.2] ])
        predicted_uncertinty_0 = np.array([ [k*math.pow(sigma_x_y[0],2),0,0], [0,k*math.pow(2,2),0], [0,0,1.2] ])

        X_Y_theta_est , uncertinty_cov_list = ExtendedKalmanFilter.performe_KalmanFilter(ekf,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n, EFK = True, vf_array = vf_array_noised, wz_array = wz_array_noised, sigma_vf = 2, sigma_wz = 0.2, yaw_rate_and_vf_noised = True)         
        RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(X_Y_theta_GT,X_Y_theta_est)
        print("RMSE", RMSE, "maxE", maxE)
        #print("theta est list:",X_Y_theta_est[:,2] )
        graphs.plot_single_graph(X_Y_theta_est[:,2],'est yaw angles', 'time elapsed [sec]','yaw angle [rad]','yaw angles GT' )

        #print("RMSE " ,RMSE_list[np.argmin(maxE_list)] ,"maxE" , min(maxE_list), 'k' , k)
        X_Y_est = X_Y_theta_est[:,:2]

        # build_ENU_from_GPS_trajectory
        graphs.plot_three_graphs( self.ENU_locations_array, X_Y_theta_est,self.ENU_locations_array_noised, 'EKF results with noise in commands:' , 'X [m]', 'Y [m]','GT trajectory', 'estimated trajectory', 'observed trajectory')
        
        # make dead reckoning after 5 seconds (inserting dead reckoning = true):
        EKF_dead_reckoning = ExtendedKalmanFilter()
        X_Y_theta_est_dead_reckoning , uncertinty_cov_list_dead_reckoning = ExtendedKalmanFilter.performe_KalmanFilter(ekf,predicted_mean_0,predicted_uncertinty_0,observation_z_t_array,self.times_array, sigma_x_y, sigma_n , EFK = True, vf_array = vf_array_noised, wz_array = wz_array_noised, sigma_vf = 2, sigma_wz = 0.2, yaw_rate_and_vf_noised = True, dead_reckoning = True) 
        X_Y_est_dead_reckoning = X_Y_theta_est_dead_reckoning[:,:2]
        
        # make animation from gt est and est dead reckoning:
        X_Y_GT_locations = self.ENU_locations_array[:,:2]

        #print("uncertinty_cov_list shape", uncertinty_cov_list.shape, "uncertinty_cov_list",uncertinty_cov_list)
        X_XY_XY_Y_uncertinty_cov_list = uncertinty_cov_list[:,[0,1,3,4]]
        print("X_XY_XY_Y_uncertinty_cov_list.shape:" ,X_XY_XY_Y_uncertinty_cov_list.shape,"X_XY_XY_Y_uncertinty_cov_list",X_XY_XY_Y_uncertinty_cov_list)
        ani = graphs.build_animation(X_Y_GT_locations, X_Y_est, X_Y_est_dead_reckoning, X_XY_XY_Y_uncertinty_cov_list, 'trajectories', 'X [m]', 'Y [m]', 'GT', 'EKF_estimat', 'dead reckoning')
        graphs.save_animation(ani, basedir, 'animation of GT EKF estimate and dead reckoning')
        

        # Plot and analyze the estimated x-y values separately and corresponded
        # sigma value along the trajectory. (e.g. show in same graph xestimated-xGT and
        # 𝜎𝑥 values and explain your results):
        
        X_Y_theta_estimated_minus_X_Y_theta_GT = X_Y_theta_est - X_Y_theta_GT
        times_array = self.times_array.reshape((self.times_array.shape[0],1))

        X_estimate_minus_X_GT = X_Y_theta_estimated_minus_X_Y_theta_GT[:,0].reshape((X_Y_theta_estimated_minus_X_Y_theta_GT[:,0].shape[0],1))
        X_estimate_minus_X_GT_and_times = np.concatenate((times_array,X_estimate_minus_X_GT),axis =1)
        sigma_x = X_XY_XY_Y_uncertinty_cov_list[:,0]
        sigma_x = np.reshape(sigma_x,(sigma_x.shape[0],1))
        sigma_minus_x = (-sigma_x)
        sigma_x_with_times = np.concatenate((times_array, sigma_x),axis =1)
        sigma_minus_x_with_times = np.concatenate((times_array, sigma_minus_x),axis =1)
        graphs.plot_two_graphs_one_double(X_estimate_minus_X_GT_and_times,sigma_x_with_times,sigma_minus_x_with_times,'X estimated - XGT and sigma_x values', 'Times elapsed [sec]', 'X estimation error [m]' ,'estimation eror', 'estimated 1 sigma interval')
        
        # calc sigma y grapgh
        Y_estimate_minus_Y_GT = X_Y_theta_estimated_minus_X_Y_theta_GT[:,1].reshape((X_Y_theta_estimated_minus_X_Y_theta_GT[:,1].shape[0],1))
        Y_estimate_minus_Y_GT_and_times = np.concatenate((times_array,Y_estimate_minus_Y_GT),axis =1)
        sigma_y = X_XY_XY_Y_uncertinty_cov_list[:,3]
        sigma_y = np.reshape(sigma_y,(sigma_y.shape[0],1))
        sigma_minus_y = (-sigma_y)
        sigma_y_with_times = np.concatenate((times_array, sigma_y),axis =1)
        sigma_minus_y_with_times = np.concatenate((times_array, sigma_minus_y),axis =1)
        graphs.plot_two_graphs_one_double(Y_estimate_minus_Y_GT_and_times,sigma_y_with_times,sigma_minus_y_with_times,'Y estimated - YGT and sigma_y values', 'Times elapsed [sec]', 'Y estimation error [m]' ,'estimation eror', 'estimated 1 sigma interval')
        
        # calc sigma theta grapgh theta = yaw
      
        theta_estimate_minus_theta_GT = X_Y_theta_estimated_minus_X_Y_theta_GT[:,2].reshape((X_Y_theta_estimated_minus_X_Y_theta_GT[:,2].shape[0],1))
        index_theta_estimate_minus_theta_GT = theta_estimate_minus_theta_GT > np.pi
        theta_estimate_minus_theta_GT[index_theta_estimate_minus_theta_GT] -= 2*np.pi
        index_theta_estimate_minus_theta_GT = theta_estimate_minus_theta_GT < -np.pi
        theta_estimate_minus_theta_GT[index_theta_estimate_minus_theta_GT] += 2*np.pi
        theta_estimate_minus_theta_GT_and_times = np.concatenate((times_array,theta_estimate_minus_theta_GT),axis =1)
        sigma_theta = uncertinty_cov_list[:,[8]]
        sigma_theta = np.reshape(sigma_theta,(sigma_theta.shape[0],1))
        sigma_minus_theta = (-sigma_theta)
        sigma_theta_with_times = np.concatenate((times_array, sigma_theta),axis =1)
        sigma_minus_theta_with_times = np.concatenate((times_array, sigma_minus_theta),axis =1)
        graphs.plot_two_graphs_one_double(theta_estimate_minus_theta_GT_and_times,sigma_theta_with_times,sigma_minus_theta_with_times,'theta estimated - theta GT and sigma_theta values', 'Times elapsed [sec]', 'theta estimation error [rad]' ,'estimation eror', 'estimated 1 sigma interval')
        
        
        '''
        # sigma_samples = 
        
        # sigma_vf, sigma_omega = 
        
        # build_LLA_GPS_trajectory
        
        # add_gaussian_noise to u and measurments (locations_gt[:,i], sigma_samples[i])
            
        # ekf = ExtendedKalmanFilter(sigma_samples, sigma_vf, sigma_omega)
        # locations_ekf, sigma_x_xy_yx_y_t = ekf.run(locations_noised, times, yaw_vf_wz_noised, do_only_predict=False)
        
        # RMSE, maxE = ekf.calc_RMSE_maxE(locations_gt, locations_ekf)
 
        # build_animation
        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")
        '''
    
    def Q3(self,basedir):

        landmarks = self.dataset.load_landmarks()
        sensor_data_gt = self.dataset.load_sensor_data()
        #print("landmarks.shape" ,landmarks.shape)
        print("landmarks:" ,landmarks)
        #print("sensor_data_gt.shape", sensor_data_gt.shape)
        print("sensor_data_gt:", sensor_data_gt)

        sigma_x_y_theta =[2,2, 0.6]# [0.1,0.1, 0.02] #TODO
        variance_r1_t_r2 = [0.01,0.1,0.01]#TODO
        
        variance_r_phi = [0.1,0.01] #TODO
        
        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)
        
        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)
        
        graphs.plot_single_graph(mu_arr_gt, 'odometry GT trajectory', 'X [m]', 'Y [m]', 'GT trajectory odometry' )
        maxE = 0
        e_x = mu_arr_gt[20:,0] - mu_arr[20:,0]
        e_y = mu_arr_gt[20:,1] - mu_arr[20:,1]
        maxE = max(abs(e_x)+abs(e_y))
        RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(mu_arr_gt.shape[0]-20))
        print("RMSE", RMSE, "maxE", maxE)
        graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-$x_n$", "frame", "error", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
        graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-$y_n$", "frame", "error", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
        graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "$\\theta-\\theta_n$", 
                                 "frame", "error", "$\\theta-\\theta_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))
        
        graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]), 
                                 "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
        graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]), 
                                 "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))
        
        graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
                                 "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
        graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
                                 "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))
        
        ax.set_xlim([-2, 12])
        ax.set_ylim([-2, 12])
        
        from matplotlib import animation
        ani = animation.ArtistAnimation(fig, frames, repeat=False)
        graphs.show_graphs()
        #ani.save('im.mp4', metadata={'artist':'me'})
        graphs.save_animation(ani, basedir, 'animation of Trajectory of EKF-SLAM')
    '''
    def run(self):
        self.Q1()
    '''   
        
