import numpy as np
import math
import matplotlib.pyplot as plt
#from utils.plot_state import plot_state
from plot_state import plot_state

from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:

  #TODO
  def one_step_of_KalmanFilter(self,previus_mean_t_minus_1, 
                                  uncertinty_of_previus_belef_t_minus_1,
                                  observation_z_t,
                                  delta_t,
                                  sigma_x_y,
                                  sigma_n,
                                  const_acc = False,
                                  EKF = False,
                                  yaw_rate_and_vf_noised = False,
                                  vf_t = None,
                                  wz_t = None,
                                  sigma_vf = 1,
                                  sigma_wz = 1,
                                  set_kalman_gain_to_zero = False,
                                  control_command_t = None ):
      
      A_t = np.array([[1,delta_t,0,0],[0,1,0,0],[0,0,1,delta_t],[0,0,0,1]])
      C_t = np.array([[1,0,0,0],[0,0,1,0]])
      R_t = np.array([[0,0,0,0],[0,delta_t,0,0],[0,0,0,0],[0,0,0,delta_t]])*math.pow(sigma_n,2)
      Q_t = np.array([[sigma_x_y[0]*sigma_x_y[0],0],[0,sigma_x_y[1]*sigma_x_y[1]]])
      B_t = None
      if(const_acc == True):
        A_t = np.array([[1,delta_t,np.power(delta_t,2)/2,0,0,0],[0,1,delta_t,0,0,0],[0,0,1,0,0,0],[0,0,0,1,delta_t,np.power(delta_t,2)/2],[0,0,0,0,1,delta_t],[0,0,0,0,0,1]])
        C_t = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
        R_t = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,delta_t,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,delta_t]])*math.pow(sigma_n,2)
        Q_t = np.array([[sigma_x_y[0]*sigma_x_y[0],0],[0,sigma_x_y[1]*sigma_x_y[1]]])
        B_t = None
      
      # predictin step:
      
      if (EKF == True):
        #previus_mean_t_minus_1[2][0] = np.clip(previus_mean_t_minus_1[2][0], a_min = - math.pi, a_max = math.pi)
        if(previus_mean_t_minus_1[2][0] > (math.pi)):
          previus_mean_t_minus_1[2][0] = previus_mean_t_minus_1[2][0] - 2*math.pi
        if(previus_mean_t_minus_1[2][0] < - (math.pi)):
          previus_mean_t_minus_1[2][0] = previus_mean_t_minus_1[2][0] + 2*math.pi
        
        #  g(control_command_t,previus_mean_t_minus_1)
        # A_t is G_T
        yaw_t_minus_1 = previus_mean_t_minus_1[2][0]
        v_cos_theta_devided_by_w = float(vf_t*np.cos(yaw_t_minus_1)/wz_t)
        v_cos_theta_plus_w_delta_t_devided_by_w = float(vf_t*np.cos(yaw_t_minus_1 + wz_t*delta_t)/wz_t)
        v_sin_theta_devided_by_w = float(vf_t*np.sin(yaw_t_minus_1)/wz_t)
        v_sin_theta_plus_w_delta_t_devided_by_w = float(vf_t*np.sin(yaw_t_minus_1 + wz_t*delta_t)/wz_t)
        
        C_t = np.array([[1,0,0],[0,1,0]])
        V_t = np.array([[ -v_sin_theta_devided_by_w/vf_t + v_sin_theta_plus_w_delta_t_devided_by_w/vf_t, (v_sin_theta_devided_by_w - v_sin_theta_plus_w_delta_t_devided_by_w)/wz_t + v_cos_theta_plus_w_delta_t_devided_by_w*delta_t ], [v_cos_theta_devided_by_w/vf_t - v_cos_theta_plus_w_delta_t_devided_by_w/vf_t, (-v_cos_theta_devided_by_w + v_cos_theta_plus_w_delta_t_devided_by_w)/wz_t + v_sin_theta_plus_w_delta_t_devided_by_w*delta_t], [0,delta_t]],dtype = float)
                        
        #print("V_t shape:", V_t.shape , V_t)
        A_t  = np.array([[1, 0, -v_cos_theta_devided_by_w + v_cos_theta_plus_w_delta_t_devided_by_w ], [0, 1, -v_sin_theta_devided_by_w + v_sin_theta_plus_w_delta_t_devided_by_w], [0,0,1]], dtype = float)  
        if (yaw_rate_and_vf_noised):                       
          R_t_top =  np.array([[sigma_vf*sigma_vf,0],[0, sigma_wz*sigma_wz]]) 
                       
        else:
          R_t_top = np.array([[0,0],[0,0]])
        R_t = np.dot(V_t,np.dot(R_t_top,V_t.T))  +  np.array([[0,0,0],[0,0,0],[0,0,delta_t]])*sigma_n 
        predicted_mean_t = previus_mean_t_minus_1 + np.array([-vf_t*np.sin(yaw_t_minus_1)/wz_t + vf_t*np.sin(yaw_t_minus_1 + wz_t*delta_t)/wz_t, vf_t*np.cos(yaw_t_minus_1)/wz_t - vf_t*np.cos(yaw_t_minus_1 + wz_t*delta_t)/wz_t, normalize_angle(wz_t*delta_t)])

      else:
      
        predicted_mean_t = np.dot(A_t,previus_mean_t_minus_1) # + np.matmul(B_t,control_command_t) 

      
      predicted_uncertinty_t = np.matmul(A_t, np.matmul(uncertinty_of_previus_belef_t_minus_1,A_t.T)) + R_t

      # correction step:
      if set_kalman_gain_to_zero == True:
        kalman_gain_t = np.zeros((4,2))
        if EKF== True:
          kalman_gain_t = np.zeros((3,2))
      else:
        kalman_gain_input = np.matmul(C_t, np.matmul(predicted_uncertinty_t, C_t.T)) + Q_t
        #kalman_gain_input = np.array(kalman_gain_input)
        #print (" kalman_gain_input shape " , kalman_gain_input.shape, kalman_gain_input)
        kalman_gain_t = np.matmul(predicted_uncertinty_t,np.matmul(C_t.T,np.linalg.inv(kalman_gain_input)))  
      corrected_mean_t = predicted_mean_t +np.matmul(kalman_gain_t,(observation_z_t-np.matmul(C_t, predicted_mean_t)))
      if EKF == True:
        corrected_mean_t[2] = normalize_angle(corrected_mean_t[2])

      I = np.identity(predicted_mean_t.shape[0])
      corrected_uncertinty_t = np.matmul(I -np.matmul(kalman_gain_t,C_t),predicted_uncertinty_t)
      corrected_uncertinty_t[2][2] = np.clip(corrected_uncertinty_t[2][2], a_min = 0,a_max = 2*math.pi)
      return corrected_mean_t , corrected_uncertinty_t


  def performe_KalmanFilter(self,predicted_mean_0, predicted_uncertinty_0, observation_z_t_array,times_array ,sigma_x_y, sigma_n,const_acc = False, EFK = False,
                                  yaw_rate_and_vf_noised = False,
                                  vf_array = None,
                                  wz_array = None,
                                  sigma_vf = 1,
                                  sigma_wz = 1,
                                  dead_reckoning = False):
    X_Y_est = []
    previus_mean_t_minus_1 = predicted_mean_0.T.reshape(predicted_mean_0.T.shape[0],1)
    
    X_Y_est.append(np.squeeze(predicted_mean_0))
    uncertinty_cov_list = []
    uncertinty_of_previus_belef_t_minus_1 = predicted_uncertinty_0
    uncertinty_cov_list.append(np.squeeze(uncertinty_of_previus_belef_t_minus_1).flatten())
    set_kalman_gain_to_zero = False
    vf_t = None
    wz_t = None 
    for i in range(observation_z_t_array.shape[0]-1):
      #for i in range(4):
      delta_t = times_array[i+1] - times_array[i]
      #print("times_array[i]", times_array[i])
      if (dead_reckoning and (times_array[i] > 5)) :
        set_kalman_gain_to_zero = True
      if(EFK == True):
        vf_t = vf_array[i]
        wz_t = wz_array[i]

      observation_z_t = observation_z_t_array[i+1].T.reshape((2,1))
      corrected_mean_i , corrected_uncertinty_i = self.one_step_of_KalmanFilter(previus_mean_t_minus_1,uncertinty_of_previus_belef_t_minus_1, observation_z_t, delta_t, sigma_x_y, sigma_n,const_acc, EFK, yaw_rate_and_vf_noised, vf_t, wz_t, sigma_vf, sigma_wz, set_kalman_gain_to_zero)
      #print("corrected_mean_i shape " , corrected_mean_i)
      #print("corrected_uncertinty_i shape", corrected_uncertinty_i)
      X_Y_est.append(np.squeeze(corrected_mean_i))
      previus_mean_t_minus_1 = corrected_mean_i
      uncertinty_of_previus_belef_t_minus_1 = corrected_uncertinty_i
      uncertinty_cov_list.append(np.squeeze(corrected_uncertinty_i).flatten())
      #print("X_Y_est", X_Y_est)
      #print("uncertinty_cov_list",uncertinty_cov_list)

    return np.array(X_Y_est) ,np.array(uncertinty_cov_list)



  #@staticmethod
  def calc_RMSE_maxE(X_Y_GT, X_Y_est):
      """
      That function calculates RMSE and maxE

      Args:
          X_Y_GT (np.ndarray): ground truth values of x and y
          X_Y_est (np.ndarray): estimated values of x and y

      Returns:
          (float, float): RMSE, maxE
      """
      maxE = 0
      e_x = X_Y_GT[100:,0] - X_Y_est[100:,0]
      e_y = X_Y_GT[100:,1] - X_Y_est[100:,1]
      maxE = max(abs(e_x)+abs(e_y))
      RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2))/(X_Y_GT.shape[0]-100))





      return RMSE, maxE


class ExtendedKalmanFilter(KalmanFilter):

    #TODO

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
      """
      That function calculates RMSE and maxE

      Args:
          X_Y_GT (np.ndarray): ground truth values of x and y
          X_Y_est (np.ndarray): estimated values of x and y

      Returns:
          (float, float): RMSE, maxE
      """
      maxE = 0
      e_x = X_Y_GT[100:,0] - X_Y_est[100:,0]
      e_y = X_Y_GT[100:,1] - X_Y_est[100:,1]
      e_yaw = X_Y_GT[100:,2] - X_Y_est[100:,2]
      abs_e_yaw = abs(e_yaw)
      index_e_yaw = abs_e_yaw > np.pi
      #print(index_e_yaw)
      abs_e_yaw[index_e_yaw] -= 2*np.pi
      #print(e_yaw)
      maxE = max(abs(e_x) + abs(e_y) + abs(abs_e_yaw))
      RMSE = np.sqrt(sum(np.power(e_x,2)+np.power(e_y,2) + np.power(e_yaw,2))/(X_Y_GT.shape[0]-100))
      return RMSE, maxE
  


class ExtendedKalmanFilterSLAM:
    def __init__(self, sigma_x_y_theta, variance_r1_t_r2, variance_r_phi):
        self.sigma_x_y_theta = sigma_x_y_theta  #TODO
        self.variance_r_phi = variance_r_phi   #TODO
        #self.R_x = np.array([[np.power(variance_r1_t_r2[0],2), 0, 0], [0,np.power(variance_r1_t_r2[1],2),0], [0, 0, np.power(variance_r1_t_r2[2],2)]])  #TODO cheek again!!!!!
        self.R_x = np.array([[np.power(variance_r1_t_r2[0],2), 0, 0], [0,np.power(variance_r1_t_r2[1],2),0], [0, 0, np.power(variance_r1_t_r2[2],2)]])  #TODO cheek again!!!!!

    def predict(self, mu_prev, sigma_prev, u, N):
        # Perform the prediction step of the EKF
        # u[0]=translation, u[1]=rotation1, u[2]=rotation2
      
        delta_trans, delta_rot1, delta_rot2 = u['t'], u['r1'], u['r2']    #TODO
        #print("mu_prev", mu_prev.shape)
        #print(mu_prev[2])
        theta_prev = normalize_angle(mu_prev[2])   #TODO 
        
        F = np.hstack((np.identity(3),np.zeros((3,2*N))))     #TODO
        G_x = np.identity(3) + np.array([[0, 0, -delta_trans*np.sin(theta_prev + delta_rot1)], [0, 0, delta_trans*np.cos(theta_prev + delta_rot1)], [0,0,0]])  #TODO jacobian of motion
        G = np.vstack((np.hstack((G_x,np.zeros((3,2*N)))) , np.hstack((np.zeros((2*N,3)),np.identity(2*N)))))    #TODO decide size of I and replace it altenitive id to add matrix rows and vetores to G_x step 4 in slideshow
        V = np.array([[-delta_trans*np.sin(theta_prev + delta_rot1), np.cos(theta_prev + delta_rot1), 0], [delta_trans*np.cos(theta_prev + delta_rot1), np.sin(theta_prev + delta_rot1),0], [1,0,1]])  #TODO
        R_hat_x = np.dot(V,np.dot(self.R_x,V.T)) + np.array([[0,0,0],[0,0,0],[0,0,1.3]])
        mu_est = mu_prev + np.dot(F.T, np.array([ delta_trans * np.cos(theta_prev + delta_rot1), delta_trans * np.sin(theta_prev + delta_rot1) , normalize_angle(delta_rot1 + delta_rot2)]).T)     #TODO step3 in slideshow
        sigma_est = np.dot(G, np.dot(sigma_prev, G.T)) + np.vstack((np.hstack((R_hat_x, np.zeros((3,2*N)))), np.zeros((2*N,2*N+3))))      #TODO  + np.dot(F.T, np.dot(R_hat_x, F))
        
        return mu_est, sigma_est
    
    def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
        # Perform filter update (correction) for each odometry-observation pair read from the data file.
        mu = mu_pred.copy()      #mu is [m_j_x,m_j_y] of all landmarks
        sigma = sigma_pred.copy()
        theta = mu[2]
        
        m = len(z["id"])
        Z = np.zeros(2 * m)
        z_hat = np.zeros(2 * m)
        H = None
        
        for idx in range(m):
            j = z["id"][idx] - 1
            r = z["range"][idx]
            phi = z["bearing"][idx]
            
            mu_j_x_idx = 3 + j*2
            mu_j_y_idx = 4 + j*2
            Z_j_x_idx = idx*2
            Z_j_y_idx = 1 + idx*2
            
            if observed_landmarks[j] == False:
                mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
                observed_landmarks[j] = True
                
            Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])
            
            delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
            q = delta.dot(delta)
            z_hat[Z_j_x_idx : Z_j_y_idx + 1] = np.array([np.sqrt(q),  normalize_angle(np.arctan2(delta[1],delta[0]) - theta)]) #.T       #TODO expected observation of landmark j
            

            I = np.diag(5*[1])
            F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))
            
            Hi = np.dot([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],[delta[1], -delta[0], -q, -delta[1], delta[0]]], F_j)/q    #TODO

            if H is None:
                H = Hi.copy()
            else:
                H = np.vstack((H, Hi))
        
        Q = np.zeros((H.shape[0],H.shape[0]))
        np.fill_diagonal(Q, [np.power(self.variance_r_phi[0],2),np.power(self.variance_r_phi[1],2)] ) #TODO
            
        
        #print("sigma_pred ",sigma_pred.shape)
        #print("H:" , H.shape)  
        #print("hi", Hi.shape)
        S = np.linalg.inv(np.dot(H,np.dot(sigma_pred,H.T)) + Q) #TODO
        K = np.dot(sigma_pred,np.dot(H.T,S)) #
        #print("k", K.shape)
        
        #print("Z" , Z.shape)
        #print("z_hat", z_hat.shape)
        diff =  Z - z_hat   #TODO
       
        diff[1::2] = normalize_angles_array(diff[1::2])
        
        mu = mu + K.dot(diff)
        sigma = np.dot((np.identity(2*N+3) - np.dot(K,H)),sigma_pred)  #TODO uncertinty matrix of full cov
        
        mu[2] = normalize_angle(mu[2])

        # Remember to normalize the bearings after subtracting!
        # (hint: use the normalize_all_bearings function available in tools)

        # Finish the correction step by computing the new mu and sigma.
        # Normalize theta in the robot pose.

        
        return mu, sigma, observed_landmarks
    
    def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
        # Get the number of landmarks in the map
        N = len(landmarks)
        
        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution


        init_inf_val = 100 #TODO
        
        #mu_arr = [np.hstack((self.sigma_x_y_theta,np.zeros(2*N))).T] #TODO
       
        mu_arr = [np.hstack(([0.096,0.0101,0.01009],np.zeros(2*N))).T] #TODO
        #mu_arr = [np.hstack(([0,0,0],np.zeros(2*N))).T]
        sigma_prev = np.vstack((np.hstack(([[np.power(self.sigma_x_y_theta[0],2), 0, 0], [0, np.power(self.sigma_x_y_theta[1],2), 0], [0, 0, np.power(self.sigma_x_y_theta[2],2)]],np.zeros((3,2*N)))), np.hstack((np.zeros((2*N,3)),init_inf_val*np.identity(2*N)))))   #TODO

        # sigma for analysis graph sigma_x_y_t + select 2 landmarks
        landmark1_ind= 3    #TODO
        landmark2_ind= 4 #TODO

        Index=[0,1,2,landmark1_ind,landmark1_ind+1,landmark2_ind,landmark2_ind+1]
        sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index,Index].copy()
        
        observed_landmarks = np.zeros(N, dtype=bool)
        
        sensor_data_count = int(len(sensor_data_noised) / 2)
        frames = []
        
        mu_arr_gt = np.array([[0, 0, 0]])

        for idx in range(sensor_data_count):
            mu_prev = mu_arr[-1]
            
            u = sensor_data_noised[(idx, "odometry")]
            # predict
            mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)
            # update (correct)
            mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)
            
            mu_arr = np.vstack((mu_arr, mu))
            sigma_prev = sigma.copy()
            sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index,Index].copy()))
            
            delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
            delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
            delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]

            calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
            calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)

            theta = delta_r1_gt + delta_r2_gt

            theta_prev = mu_arr_gt[-1,2]
            mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))
            
            frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])
            
            frames.append(frame)
        
        return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2
  