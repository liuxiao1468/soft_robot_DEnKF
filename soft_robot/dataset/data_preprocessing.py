import numpy as np
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



def main():
    index = ['time', 'desired', 'current', 'imus_acc', 'imus_vel', 'mocap_pos', 'mocap_rot']
    dataset = pickle.load(open('dataset_4.pkl', 'rb'))

    # actions
    action = dataset['desired']
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


    # with open('parameter_4.pkl', 'wb') as handle:
    #     pickle.dump(parameters, handle)

    # #########create dataset for the filter - train #########
    # num_points = 700000
    # state_pre = state[0:num_points]
    # state_gt = state[1:num_points+1]
    # action = action[1:num_points]
    # obs = obs[1:num_points]

    # data = dict()
    # data['state_pre'] = state_pre
    # data['state_gt'] = state_gt
    # data['action'] = action
    # data['obs'] = obs

    # with open('train_dataset_4.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    ######### create dataset for the filter - test #########
    num_points = 700000
    state_pre = state[num_points:num_points+10000]
    state_gt = state[num_points+1:num_points+10001]
    action = action[num_points:num_points+10000]
    obs = obs[num_points+1:num_points+10001]

    data = dict()
    data['state_pre'] = state_pre
    data['state_gt'] = state_gt
    data['action'] = action
    data['obs'] = obs

    with open('test_dataset_4.pkl', 'wb') as f:
        pickle.dump(data, f)

    # data_tmp = pickle.load(open('test_dataset_4.pkl', 'rb'))
    # print(len(data_tmp['state_gt']))
    # print(data_tmp['obs'][24705])
    # print(data_tmp['obs'][24704])
    

    



    # ######### Creating figure #########
    # fig = plt.figure(figsize = (10, 7))

    # # the pressure vector
    # tmp_arr = action[400:500,:].T
    # plt.imshow(tmp_arr, interpolation='nearest')


    # # # the IMU readings
    # # x = np.linspace(1, 10000, 10000)
    # # for i in range (15):
    # #     plt.plot(x, obs[:x.shape[0],i].flatten(), '--' ,linewidth=1, alpha=0.5)

    # # # the 3d plot for the state
    # # ax = plt.axes(projection ="3d")
    # # # Creating plot
    # # ax.scatter3D(state[:20000,0], state[:20000,1], state[:20000,2])


    # plt.title("pressure vector - actions")
    # # show plot
    # plt.show()

    

    

if __name__ == '__main__':
    main()