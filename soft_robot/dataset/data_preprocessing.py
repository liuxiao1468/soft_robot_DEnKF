import numpy as np
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import re
import random
import roslibpy

HEADER = 2

def read_tmbagcsv(filename):
      print("Reading a tmbag csv file...")
      df = pd.read_csv(filename, low_memory=False)

      #===========================================
      print("Decoding imu signals...")
      imus_acc = []
      tags = [ "port_dev_" + s for s in re.findall(r"\'([^\']*)\'", df['/suppl/notes.1'][HEADER]) ]
      for tag in tags:
            tmp =  df['/nanorp_imu/'+tag+'.12']
            tmp = tmp[HEADER:]   # remove the variable names
            x = np.vectorize(float)(tmp)
            tmp =  df['/nanorp_imu/'+tag+'.13']
            tmp = tmp[HEADER:]   # remove the variable names
            y = np.vectorize(float)(tmp)
            tmp =  df['/nanorp_imu/'+tag+'.14']
            tmp = tmp[HEADER:]   # remove the variable names
            z = np.vectorize(float)(tmp)
            imus_acc.append(np.vstack((np.array(x),np.array(y),np.array(z))).T)

      # tags = [ "ip"+s.replace('.','_') for s in re.findall(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}', df['/suppl/notes.1'][1])]
      # for tag in tags:
      #       tmp =  df['/nanorp_imu/'+tag+'.1']
      #       tmp = tmp[1:]   # remove the variable name
      #       tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', l) for l in tmp])
      #       tmp = np.vectorize(float)(tmp)
      #       imus_acc.append(tmp)

      imus_vel = []
      for tag in tags:
            tmp =  df['/nanorp_imu/'+tag+'.7']
            tmp = tmp[HEADER:]   # remove the variable names
            x = np.vectorize(float)(tmp)
            tmp =  df['/nanorp_imu/'+tag+'.8']
            tmp = tmp[HEADER:]   # remove the variable names
            y = np.vectorize(float)(tmp)
            tmp =  df['/nanorp_imu/'+tag+'.9']
            tmp = tmp[HEADER:]   # remove the variable names
            z = np.vectorize(float)(tmp)
            imus_vel.append(np.vstack((np.array(x),np.array(y),np.array(z))).T)

      # for tag in tags:
      #       tmp =  df['/nanorp_imu/'+tag+'.2']
      #       tmp = tmp[1:]   # remove the variable name
      #       tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', l) for l in tmp])
      #       tmp = np.vectorize(float)(tmp)
      #       imus_vel.append(tmp)

      #===========================================
      print("Decoding desired pressure signals...")
      board = []
      for i in range(4):
            tmp =  df['/tenpa/pressure/desired%d.1' % i]
            tmp = tmp[HEADER:]   # remove the variable name
            tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\df+)+', l) for l in tmp])
            tmp = np.vectorize(int)(tmp)
            board.append(tmp[:,0:10]) # remove ch.11 and ch.12 and put into a list

      desired = np.concatenate(board, 1)

      #===========================================
      print("Decoding current pressure signals...")
      board = []
      for i in range(4):
            tmp =  df['/tenpa/pressure/current%d.1' % i]
            tmp = tmp[HEADER:]   # remove the variable name
            tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\df+)+', l) for l in tmp])
            tmp = np.vectorize(int)(tmp)
            board.append(tmp[:,0:10]) # remove ch.11 and ch.12 and put into a list

      current = np.concatenate(board, 1)

      #===========================================
      print("Decoding mocap signals...")
      tmp =  df['/mocap/rigidbody1.2']
      tmp = tmp[HEADER:]   # remove the variable name
      tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', l) for l in tmp])
      mocap_pos = np.vectorize(float)(tmp)

      tmp =  df['/mocap/rigidbody1.3']
      tmp = tmp[HEADER:]   # remove the variable name
      tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', l) for l in tmp])
      mocap_rot = np.vectorize(float)(tmp)

      #===========================================
      print("Decoding the elasped time...")
      tmp =  df['sync_time']
      tmp = tmp[HEADER:]   # remove the variable name
      tmp = np.array([re.findall(r'[+-]?(?:\d+\.?\d*|\.\df+)+', l) for l in tmp])
      time = np.vectorize(float)(tmp)

      #===========================================
      code = df['/suppl/notes.2'][2]
      code = np.array(re.findall(r'[+-]?(?:\d+\.?\d*|\.\df+)+', code), dtype=float)


      print("Finding the cue...")
      cue = 0
      while np.all(desired[0]==desired[cue]):
            cue += 1
      cue = cue-1 # kepe the initial desired pressure

      time = time[cue:]
      desired = desired[cue:]
      current = current[cue:]
      for i in range(5):
            imus_acc[i] = imus_acc[i][cue:]
            imus_vel[i] = imus_vel[i][cue:]
      mocap_pos = mocap_pos[cue:]
      mocap_rot = mocap_rot[cue:]

      print("Done!")

      ret = {'time':time, 'desired':desired, 'current':current, 'imus_acc':imus_acc, 'imus_vel':imus_vel,
       'mocap_pos':mocap_pos, 'mocap_rot':mocap_rot, 'code':code}

      return ret

def get_data(path):
      print("****** This is a test ******")
      #df = pd.read_csv(r"..\bag2csv\bag\tmbag_33\tmbag_33.csv", low_memory=False)

      data_dict = read_tmbagcsv(path)
    #   print("time:")
    #   print(data_dict['time'].shape)
    #   print("desired:")
    #   print(data_dict['desired'].shape)
    #   print("current:")
    #   print(data_dict['current'].shape)
    #   print("There are %d IMUs" % len(data_dict['imus_acc']))
    #   for i in range(len(data_dict['imus_acc'])):
    #         print("IMU%d acc:" % i)
    #         print(data_dict['imus_acc'][i].shape)
    #         print("IMU%d vel:" % i)
    #         print(data_dict['imus_vel'][i].shape)
    #   print("mocap pos:")
    #   print(data_dict['mocap_pos'].shape)
    #   print("mocap rot:")
    #   print(data_dict['mocap_rot'].shape)
      return data_dict

def create_dataset():
    print('===========')
#     index = [33,34,35,36,37,38,39,40,41,42,43,44]
    index = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    for j in range (len(index)):
        print("----",str(index[j]))
      #   dataset = get_data('./bag_csv/tmbag_'+str(index[j])+'/tmbag_'+str(index[j])+'.csv')
        dataset = get_data('./bag_csv/tmbag_'+str(index[j])+'.csv')
        # actions
        action = dataset['current']
        print('actions: ',action.shape)

      #   # IMU readings as one array
      #   obs1_list = dataset['imus_acc']
      #   for i in range (len(obs1_list)):
      #       if i == 0:
      #           obs = obs1_list[i]
      #       else:
      #           obs = np.concatenate((obs, obs1_list[i]), axis=1)
      #   obs2_list = dataset['imus_vel']
      #   for i in range (len(obs2_list)):
      #           obs = np.concatenate((obs, obs2_list[i]), axis=1)
      #   print('observations: ',obs.shape)

        # reorganize IMUs
        obs1_list = dataset['imus_acc']
        obs2_list = dataset['imus_vel']
        for i in range (len(obs1_list)):
              imu_obs = np.concatenate((obs1_list[i], obs2_list[i]), axis=1)
              if i == 0:
                    obs = imu_obs
              else:
                    obs = np.concatenate((obs, imu_obs), axis=1)
        print('observations: ',obs.shape)
        
        # mocap as array
        state = dataset['mocap_pos']
        ori = dataset['mocap_rot']
        state = np.concatenate((state, ori), axis=1)
        print('observations: ',obs.shape)
        print('states: ',state.shape)


        # code
        code = dataset['code']
        code = np.tile(code, (len(dataset['current']), 1))
        print('code shape',code.shape)

        for i in range (state.shape[0]):
              if abs(state[i][0])>1500 or  abs(state[i][1])>1500 or abs(state[i][2])>1500 or state[i][2]<-400:
                    state[i] = state[i-1]
                    obs[i] = obs[i-1]
                    action[i] = action[i-1]

        parameters = dict()
        action_m = np.mean(action, axis = 0)
        action_std = np.std(action, axis = 0)
        obs_m = np.mean(obs, axis = 0)
        obs_std = np.std(obs, axis = 0)
        state_m = np.mean(state, axis = 0)
        state_std = np.std(state, axis = 0)
        parameters['action_m'] = action_m
        parameters['action_std'] = action_std
        parameters['obs_m'] = obs_m
        parameters['obs_std'] = obs_std
        parameters['state_m'] = state_m
        parameters['state_std'] = state_std


        with open('./processed_data/parameter_'+str(index[j])+'.pkl', 'wb') as handle:
            pickle.dump(parameters, handle)

        #########create dataset for the filter - train #########
        num_points = int(state.shape[0]*0.8)-1
        state_pre = state[0:num_points]
        state_gt = state[1:num_points+1]
        action_gt = action[1:num_points+1]
        obs_gt = obs[1:num_points+1]
        code_gt = code[1:num_points+1]
        print('code gt shape, ',code_gt.shape)

        data = dict()
        data['state_pre'] = state_pre
        data['state_gt'] = state_gt
        data['action'] = action_gt
        data['obs'] = obs_gt
        data['code'] = code_gt

        with open('./processed_data/train_dataset_'+str(index[j])+'.pkl', 'wb') as f:
            pickle.dump(data, f)

        ######### create dataset for the filter - test #########
        state_pre = state[num_points:-1]
        state_gt = state[num_points+1:]
        action_gt = action[num_points:]
        obs_gt = obs[num_points+1:]
        code_gt = code[num_points+1:]

        data = dict()
        data['state_pre'] = state_pre
        data['state_gt'] = state_gt
        data['action'] = action_gt
        data['obs'] = obs_gt
        data['code'] = code_gt

        with open('./processed_data/test_dataset_'+str(index[j])+'.pkl', 'wb') as f:
            pickle.dump(data, f)

def create_dataset_four():
      dataset = pickle.load(open('dataset_4.pkl', 'rb'))
      # actions
      action = dataset['current']
      print('actions: ',action.shape)
      # IMU readings as one array
      obs1_list = dataset['imus_acc']
      for i in range (len(obs1_list)):
            if i == 0:
                  obs = obs1_list[i]
            else:
                  obs = np.concatenate((obs, obs1_list[i]), axis=1)
      obs2_list = dataset['imus_vel']
      for i in range (len(obs2_list)):
            obs = np.concatenate((obs, obs2_list[i]), axis=1)
      print('observations: ',obs.shape)

      # mocap as array
      state = dataset['mocap_pos']
      print('states: ',state.shape)

      parameters = dict()
      action_m = np.mean(action, axis = 0)
      action_std = np.std(action, axis = 0)
      obs_m = np.mean(obs, axis = 0)
      obs_std = np.std(obs, axis = 0)
      state_m = np.mean(state, axis = 0)
      state_std = np.std(state, axis = 0)
      parameters['action_m'] = action_m
      parameters['action_std'] = action_std
      parameters['obs_m'] = obs_m
      parameters['obs_std'] = obs_std
      parameters['state_m'] = state_m
      parameters['state_std'] = state_std


      with open('./processed_data/parameter_4.pkl', 'wb') as handle:
            pickle.dump(parameters, handle)

      #########create dataset for the filter - train #########
      num_points = 500000
      state_pre = state[0:num_points]
      state_gt = state[1:num_points+1]
      action_gt = action[1:num_points]
      obs_gt = obs[1:num_points]

      data = dict()
      data['state_pre'] = state_pre
      data['state_gt'] = state_gt
      data['action'] = action_gt
      data['obs'] = obs_gt

      with open('./processed_data/train_dataset_4.pkl', 'wb') as f:
            pickle.dump(data, f)

      ######### create dataset for the filter - test #########
      state_pre = state[num_points:num_points+10000]
      state_gt = state[num_points+1:num_points+10001]
      action_gt = action[num_points:num_points+10000]
      obs_gt = obs[num_points+1:num_points+10001]

      data = dict()
      data['state_pre'] = state_pre
      data['state_gt'] = state_gt
      data['action'] = action
      data['obs'] = obs

      with open('./processed_data/test_dataset_4.pkl', 'wb') as f:
            pickle.dump(data, f)


def create_merged_dataset():
      index = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
      for j in range (len(index)):
            print("----",str(index[j]))
            dataset = get_data('./bag_csv/tmbag_'+str(index[j])+'.csv')
            # actions
            action = dataset['current']
            print('actions: ',action.shape)

            # reorganize IMUs
            obs1_list = dataset['imus_acc']
            obs2_list = dataset['imus_vel']
            for i in range (len(obs1_list)):
                  imu_obs = np.concatenate((obs1_list[i], obs2_list[i]), axis=1)
                  if i == 0:
                        obs = imu_obs
                  else:
                        obs = np.concatenate((obs, imu_obs), axis=1)
            print('observations: ',obs.shape)
            
            # mocap as array
            state = dataset['mocap_pos']
            ori = dataset['mocap_rot']
            state = np.concatenate((state, ori), axis=1)
            print('observations: ',obs.shape)
            print('states: ',state.shape)


            for i in range (state.shape[0]):
                  if abs(state[i][0])>1500 or  abs(state[i][1])>1500 or abs(state[i][2])>1500 or state[i][2]<-400:
                        state[i] = state[i-1]
                        obs[i] = obs[i-1]
                        action[i] = action[i-1]

            # code
            code = dataset['code']
            code = np.tile(code, (len(dataset['current']), 1))

            num_points = int(state.shape[0]*0.8)-1
            state_gt = state[1:num_points+1]
            action = action[1:num_points+1]
            obs = obs[1:num_points+1]
            code = code[1:num_points+1]
            state_pre = state[0:num_points]

            if j == 0:
                  merge_state_pre = state_pre
                  merge_state_gt = state_gt
                  merge_observtion = obs
                  merge_action = action
                  merge_code = code
            else:
                  merge_state_pre = np.concatenate((merge_state_pre, state_pre), axis=0)
                  merge_state_gt = np.concatenate((merge_state_gt, state_gt), axis=0)
                  merge_observtion = np.concatenate((merge_observtion, obs), axis=0)
                  merge_action = np.concatenate((merge_action, action), axis=0)
                  merge_code = np.concatenate((merge_code, code), axis=0)

      state_pre = merge_state_pre
      state_gt = merge_state_gt
      obs = merge_observtion
      action = merge_action

      parameters = dict()
      action_m = np.mean(action, axis = 0)
      action_std = np.std(action, axis = 0)
      obs_m = np.mean(obs, axis = 0)
      obs_std = np.std(obs, axis = 0)
      state_m = np.mean(state_gt, axis = 0)
      state_std = np.std(state_gt, axis = 0)
      parameters['action_m'] = action_m
      parameters['action_std'] = action_std
      parameters['obs_m'] = obs_m
      parameters['obs_std'] = obs_std
      parameters['state_m'] = state_m
      parameters['state_std'] = state_std


      with open('./processed_data/parameter_merge.pkl', 'wb') as handle:
            pickle.dump(parameters, handle)

      #########create dataset for the filter - train #########
      data = dict()
      data['state_pre'] = state_pre
      data['state_gt'] = state_gt
      data['action'] = action
      data['obs'] = obs
      data['code'] = merge_code

      with open('./processed_data/train_dataset_merge.pkl', 'wb') as f:
            pickle.dump(data, f)


def main():
      # create_dataset()
      # create_merged_dataset()
      data = pickle.load(open('./processed_data/test_dataset_60.pkl', 'rb'))
      print(data['state_pre'].shape)
      print(data['obs'].shape)
      print(data['action'].shape)
      print(data['obs'][100])
      print(data['state_pre'][100])
      print(data['code'].shape)
      # for i in range (100):
      #       print(data['action'][i])

      # # create_dataset_four()

      # # #################### testing ground ####################
      # data_tmp = pickle.load(open('./processed_data/test_dataset_51.pkl', 'rb'))
      # print(len(data_tmp['state_gt']))
      # obs = data_tmp['obs']
      # state = data_tmp['state_gt']
      # fig = plt.figure(figsize = (10, 7))


      # # # ######### Creating figure #########
      # # # # fig = plt.figure(figsize = (10, 7))

      # # # # # the pressure vector
      # # # # tmp_arr = action[400:500,:].T
      # # # # plt.imshow(tmp_arr, interpolation='nearest')


      # # # the IMU readings
      # # x = np.linspace(1, obs.shape[0], obs.shape[0])
      # # for i in range (3):
      # #     plt.plot(x, obs[:x.shape[0],i].flatten(), '--' ,linewidth=1, alpha=0.5)

      # # the 3d plot for the state
      # ax = plt.axes(projection ="3d")
      # # Creating plot
      # ax.scatter3D(state[:,0], state[:,1], state[:,2])

      # # plt.title("pressure vector - actions")
      # # show plot
      # plt.show()

def quat_to_elev_azim_roll(q, angle_offsets=(0, 0, 0)):
    q0, q1, q2, q3 = q.w, q.x, q.y, q.z
    phi = np.arctan2(-2*q1*q2 + 2*q0*q3, q1**2 + q0**2 - q3**2 - q2**2)
    theta = np.arcsin(2*q1*q3 + 2*q0*q2)
    psi = np.arctan2(-2*q2*q3 + 2*q0*q1, q3**2 - q2**2 - q1**2 + q0**2)
    azim = np.rad2deg(phi) + angle_offsets[0]
    elev = np.rad2deg(-theta) + angle_offsets[1]
    roll = np.rad2deg(psi) + angle_offsets[2]
    return elev, azim, roll

def elev_azim_roll_to_quat(elev, azim, roll, angle_offsets=(0, 0, 0)):
    phi = np.deg2rad(azim) - angle_offsets[0]
    theta = np.deg2rad(-elev) - angle_offsets[1]
    psi = np.deg2rad(roll) - angle_offsets[2]
    q0 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q1 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) - np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    q3 = np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2)
    q = np.quaternion(q0, q1, q2, q3)
    return q



if __name__ == '__main__':
      # main()
      # modalities = [1,2,3,4,5]
      # index = random.sample(modalities, 1)
      # print(index)


      ################# ROS msg. #################
      # client = roslibpy.Ros(host='10.218.101.9', port=9090)
      # client.run()
      # print('Is ROS connected?', client.is_connected)
      # talker = roslibpy.Topic(client, '/chatter', 'std_msgs/Float64MultiArray')

      # data = pickle.load(open('./processed_data/test_dataset_60.pkl', 'rb'))
      # q_array = data['state_pre']
      # import time
      # while client.is_connected:
      #       for i in range (q_array.shape[0]):
      #             print('Sending message...')
                        
      #             q = q_array[i]
      #             q[0:3] = q[0:3]/1000
      #             talker.publish(roslibpy.Message({'data': list(q)}))

      #             time.sleep(0.1)
      ############################################
      import math

      def quaternion_to_euler(w, x, y, z):

        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        X = math.atan2(t0, t1)

        t2 = 2 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        Y = math.asin(t2)
        
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

      data = pickle.load(open('./processed_data/test_dataset_60.pkl', 'rb'))
      q_array = data['state_pre']
      q = q_array[10][-4:]
      p = q_array[10][:3]
      x,y,z = quaternion_to_euler(q[0], q[1], q[2], q[3])


      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      # Cartesian axes
      ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
      ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
      ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
      # Vector before rotation
      ax.quiver(0, 0, 0, 1, 0, 0, color='b')
      # Vector after rotation
      ax.quiver(p[0]/1000, p[1]/1000, p[2]/1000, x, y, z, color='r')
      ax.set_xlim([-1.5, 1.5])
      ax.set_ylim([-1.5, 1.5])
      ax.set_zlim([0, 3])
      plt.show()



      

      
      
      # client.terminat