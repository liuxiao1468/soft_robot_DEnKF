import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from einops import rearrange, repeat
import numpy as np
import pdb

class ProcessModel(nn.Module):
    '''
    process model takes a state or a stack of states (t-n:t-1) and 
    predict the next state t. the process model is flexiable, we can inject the known
    dynamics into it, we can also change the model architecture which takes sequential
    data as input

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_x]
    '''
    def __init__(self, num_ensemble, dim_x):
        super(ProcessModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x

        self.bayes1 = LinearFlipout(in_features=self.dim_x,out_features=64)
        self.bayes2 = LinearFlipout(in_features=64,out_features=128)
        self.bayes3 = LinearFlipout(in_features=128,out_features=64)
        self.bayes4 = LinearFlipout(in_features=64,out_features=self.dim_x)

    def forward(self, last_state):
        batch_size = last_state.shape[0]
        last_state = rearrange(last_state, 'bs k dim -> (bs k) dim', bs = batch_size, k = self.num_ensemble)
        x, _ = self.bayes1(last_state)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        update, _ = self.bayes4(x)
        state = last_state + update
        state = rearrange(state, '(bs k) dim -> bs k dim', bs = batch_size, k = self.num_ensemble)
        return state

class ObservationModel(nn.Module):
    '''
    observation model takes a predicted state at t-1 and 
    predict the corresponding oberservations. typically, the observation is part of the
    state (H as an identity matrix), unless we are using some observations indirectly to
    update the state

    input -> [batch_size, num_ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_z]
    '''
    def __init__(self, num_ensemble, dim_x, dim_z):
        super(ObservationModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.linear1 = torch.nn.Linear(self.dim_x, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, self.dim_z)

    def forward(self, state):
        batch_size = state.shape[0]
        state = rearrange(state, 'bs k dim -> (bs k) dim', bs = batch_size, k = self.num_ensemble)
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        z_pred = self.linear5(x)
        z_pred = rearrange(z_pred, '(bs k) dim -> bs k dim', bs = batch_size, k = self.num_ensemble)
        return z_pred

class BayesianSensorModel(nn.Module):
    '''
    the sensor model takes the current raw sensor (usually high-dimensional images)
    and map the raw sensor to low-dimension
    Many advanced model architecture can be explored here, i.e., Vision transformer, FlowNet,
    RAFT, and ResNet families, etc.

    input -> [batch_size, 1, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    '''
    def __init__(self, num_ensemble, dim_z):
        super(BayesianSensorModel, self).__init__()
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

        self.fc1 = nn.Linear(30, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = LinearFlipout(128, 64)
        self.fc4 = LinearFlipout(64, self.dim_z)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, 'bs k dim -> (bs k) dim')
        x = repeat(x, 'bs dim -> bs k dim', k = self.num_ensemble)
        x = rearrange(x, 'bs k dim -> (bs k) dim')
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        encoding = x
        obs, _ = self.fc4(x)
        obs = rearrange(obs, '(bs k) dim -> bs k dim', bs = batch_size, k = self.num_ensemble)
        obs_z = torch.mean(obs, axis = 1)
        obs_z = rearrange(obs_z, 'bs (k dim) -> bs k dim', k=1)
        encoding = rearrange(encoding, '(bs k) dim -> bs k dim', bs = batch_size, k = self.num_ensemble)
        encoding = torch.mean(encoding, axis = 1)
        encoding = rearrange(encoding, '(bs k) dim -> bs k dim', bs = batch_size, k = 1)
        return obs, obs_z, encoding

class ObservationNoise(nn.Module):
    def __init__(self, dim_z, r_diag):
        '''
        observation noise model is used to learn the observation noise covariance matrix
        R from the learned observation, kalman filter require a explicit matrix for R
        therefore we construct the diag of R to model the noise here

        input -> [batch_size, 1, encoding/dim_z]
        output -> [batch_size, dim_z, dim_z]
        '''
        super(ObservationNoise, self).__init__()
        self.dim_z = dim_z
        self.r_diag = r_diag

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, self.dim_z)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        constant = np.ones(self.dim_z)* 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        diag = self.fc1(inputs)
        diag = F.relu(diag)
        diag = self.fc2(diag)
        diag = torch.square(diag + torch.Tensor(constant)) + torch.Tensor(init)
        diag = rearrange(diag, 'bs k dim -> (bs k) dim')
        R = torch.diag_embed(diag)
        return R

class Ensemble_KF_low(nn.Module):
    def __init__(self, num_ensemble, dim_x, dim_z):
        super(Ensemble_KF_low, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.05
        self.r_diag = self.r_diag.astype(np.float32)

        # instantiate model
        self.process_model = ProcessModel(self.num_ensemble, self.dim_x)
        self.observation_model = ObservationModel(self.num_ensemble, self.dim_x, self.dim_z)
        self.observation_noise = ObservationNoise(self.dim_z, self.r_diag)
        self.sensor_model = BayesianSensorModel(self.num_ensemble, self.dim_z)

    def forward(self, inputs, states):
        # decompose inputs and states
        batch_size = inputs.shape[0]
        raw_obs = inputs
        state_old, m_state = states

        ##### prediction step #####
        state_pred = self.process_model(state_old)
        m_A = torch.mean(state_pred, axis = 1)
        mean_A = repeat(m_A, 'bs dim -> bs k dim', k = self.num_ensemble)
        A = state_pred - mean_A
        A = rearrange(A, 'bs k dim -> bs dim k')
        
        ##### update step #####
        H_X = self.observation_model(state_pred)
        mean = torch.mean(H_X, axis = 1)
        H_X_mean = rearrange(mean, 'bs (k dim) -> bs k dim', k=1)
        m = repeat(mean, 'bs dim -> bs k dim', k = self.num_ensemble)
        H_A = H_X - m
        # transpose operation
        H_XT = rearrange(H_X, 'bs k dim -> bs dim k')
        H_AT = rearrange(H_A, 'bs k dim -> bs dim k')

        # get learned observation
        ensemble_z, z, encoding = self.sensor_model(raw_obs)
        y = rearrange(ensemble_z, 'bs k dim -> bs dim k')
        R = self.observation_noise(encoding)

        # measurement update
        innovation = (1/(self.num_ensemble -1)) * torch.matmul(H_AT, H_A) + R
        inv_innovation = torch.linalg.inv(innovation)
        K = (1/(self.num_ensemble -1)) * torch.matmul( torch.matmul(A, H_A), inv_innovation)
        gain = rearrange(torch.matmul(K, y - H_XT ), 'bs dim k -> bs k dim')
        state_new = state_pred + gain

        # gather output
        m_state_new = torch.mean(state_new, axis = 1)
        m_state_new = rearrange(m_state_new, 'bs (k dim) -> bs k dim', k=1)
        m_state_pred = rearrange(m_A, 'bs (k dim) -> bs k dim', k=1)
        output = (state_new.to(dtype=torch.float32), m_state_new.to(dtype=torch.float32), 
            m_state_pred.to(dtype=torch.float32), z.to(dtype=torch.float32), 
            ensemble_z.to(dtype=torch.float32), H_X_mean.to(dtype=torch.float32))
        return output
############ only for testing ############
# r_diag = np.ones((2)).astype(np.float32) * 0.1
# r_diag = r_diag.astype(np.float32)

# model = Ensemble_KF(32, 5, 2)
# # print('The model:')
# # print(model)
# # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print('Total number of parameters: ',pytorch_total_params)

# # some dummy input 
# state = torch.FloatTensor(8, 32, 5)
# raw_obs = torch.FloatTensor(8, 2, 224, 224)
# obs = torch.FloatTensor(8, 1, 2)

# print("check -------- ",state.dtype)

# m_state = torch.mean(state, axis = 1)

# input_state = (state, m_state)

# model(raw_obs, input_state)


