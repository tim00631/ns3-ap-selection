import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from replay_memory import PERGraphData, PrioritizedExperienceReplay, ReplayMemory, GraphReplayMemory
# from torchsummary import summary
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from model import DCRQN, GAT, GCN

class DeepQNetwork(object):
    def __init__(self,
                n_actions,
                input_size,
                device,
                timestep, # input time series feature
                batch_size,
                epsilon,
                max_timestep,
                memory_size,
                is_training,
                learning_rate=0.01,
                reward_decay=0.6,
                epsilon_min=0.01):
        self.n_actions = n_actions
        self.input_size = input_size
        self.device = device
        self.timestep = timestep
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.is_training = is_training
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.max_timestep = max_timestep
        self.diff_eps = (self.epsilon - self.epsilon_min)/self.max_timestep
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.loss_val = 0
        self.loss_history = []
        self.memory = ReplayMemory(memory_size)
        self.policy_net = DCRQN(n_actions).to(self.device)
        self.target_net = DCRQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=0.001)
            
    def choose_action(self, observation):
        # convert to Tensor so that can be calculated on GPU
        observation = torch.tensor(observation, device=self.device, dtype=torch.float)

        self.epsilon = self.epsilon - self.diff_eps
        if not self.is_training or np.random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(observation).squeeze().argmax().item()
        else:
            """random action"""
            return random.randint(0, self.n_actions-1)

    def replace_target_net_weight(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        states, nextstates, actions, rewards = self.memory.sample(self.batch_size)

        states_v = torch.tensor(states, dtype=torch.float).view(self.batch_size, 1, self.timestep, self.n_actions).to(self.device)
        actions_v = torch.tensor(actions).view(self.batch_size, 1).to(self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states_v = torch.tensor(nextstates, dtype=torch.float).view(self.batch_size, 1, self.timestep, self.n_actions).to(self.device)

        q_values = self.policy_net(states_v).squeeze()

        q_values = q_values.gather(dim=1, index=actions_v).squeeze()

        next_q_values = self.target_net(next_states_v).squeeze().view(self.batch_size, self.n_actions)
        next_q_values = next_q_values.max(1)[0].detach()
        reward_decay_v = torch.tensor(self.reward_decay, dtype=torch.float).to(self.device)

        target_value = rewards_v + reward_decay_v * next_q_values

        # print('**************************************************')
        # print('loss_values:')
        # loss_value = (q_values-target_value).detach()
        # loss_square = loss_value*loss_value
        # with open("loss_value", 'a') as f:
        #     f.write(np.array_str(loss_square.cpu().numpy().reshape(-1), max_line_width=np.inf)+'\n')
        # print(loss_square)
        # print('**************************************************')
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(q_values, target_value)
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1) ## clamp gradient to range(-1, 1)
        self.optimizer.step()
        self.loss_val = 0
        self.loss_val += loss.item()
        self.loss_history.append(self.loss_val)

class DQN4Graph():
    def __init__(self,
                n_actions,
                input_size,
                device,
                timestep, # input time series feature
                batch_size,
                epsilon,
                max_timestep,
                memory_size,
                learning_rate,
                reward_decay,
                is_training,
                epsilon_min=0.01,
                epsilon_decay=0.99):
        self.n_actions = n_actions
        self.input_size = input_size
        self.device = device
        self.timestep = timestep
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.is_training= is_training
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.max_timestep = max_timestep
        self.epsilon_decay = epsilon_decay
        self.diff_eps = (self.epsilon - self.epsilon_min)/self.max_timestep
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.loss_val = 0
        self.loss_history = []
        self.memory = GraphReplayMemory(memory_size)
        self.policy_net = GAT(timestep, n_actions, batch_size, is_training).to(self.device)
        self.target_net = GAT(timestep, n_actions, batch_size, is_training).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=0.001)
            
    def choose_action(self, observation):
        self.epsilon = self.epsilon - self.diff_eps
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if not self.is_training or np.random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(observation).squeeze().argmax().item()
        else:
            """random action"""
            return random.randint(0, self.n_actions-1)

    def replace_target_net_weight(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # def learn(self): # DQN_learn
    #     loader = DataLoader(self.memory, batch_size=self.batch_size)
    #     batch = next(iter(loader))
    #     # print(type(batch))
    #     # print(batch)
    #     states = Data(x=batch.x_s, edge_index=batch.edge_index_s) #
    #     next_states = Data(x=batch.x_t, edge_index=batch.edge_index_t)
    #     actions = batch.action.view(self.batch_size, 1).to(self.device)
    #     rewards = batch.reward.to(self.device)
    #     # print(states)
    #     # print(next_states)
    #     # print(actions.shape)
    #     # print(rewards.shape)
    #     q_values = self.policy_net(states).squeeze()
    #     # print(f'q_values.shape:{q_values.shape}')
    #     q_values = q_values.gather(dim=1, index=actions).squeeze()
    #     next_q_values = self.target_net(next_states).squeeze().view(-1, self.n_actions)
    #     next_q_values = next_q_values.max(1)[0].detach()
    #     reward_decay_v = torch.tensor(self.reward_decay, dtype=torch.float).to(self.device)
    #     target_value = rewards + reward_decay_v * next_q_values
    #     # Compute loss
    #     criterion = nn.MSELoss()
    #     loss = criterion(q_values, target_value)
    #     # Update
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     # for param in self.policy_net.parameters():
    #     #     param.grad.data.clamp_(-1, 1) ## clamp gradient to range(-1, 1)
    #     self.optimizer.step()
    #     self.loss_val = 0
    #     self.loss_val += loss.item()
    #     self.loss_history.append(self.loss_val)

    def learn(self): # Double DQN learn
        loader = DataLoader(self.memory, batch_size=self.batch_size)
        batch = next(iter(loader))
        states = Data(x=batch.x_s, edge_index=batch.edge_index_s) #
        next_states = Data(x=batch.x_t, edge_index=batch.edge_index_t)
        actions = batch.action.view(self.batch_size, 1).to(self.device)
        rewards = batch.reward.to(self.device)
        # print(states)
        # print(next_states)
        # print(actions.shape)
        # print(rewards.shape)
        q_values = self.policy_net(states).squeeze()
        # print(f'q_values.shape:{q_values.shape}')
        print(f'q_values:{q_values}')
        q_values = q_values.gather(dim=1, index=actions).squeeze()
        target_actions = self.policy_net(next_states).squeeze().view(-1, self.n_actions).max(1)[1].view(self.batch_size, 1).detach()
        next_q_values = self.target_net(next_states).squeeze().view(-1, self.n_actions).gather(dim=1, index=target_actions).squeeze()
        reward_decay_v = torch.tensor(self.reward_decay, dtype=torch.float).to(self.device)
        target_value = rewards + reward_decay_v * next_q_values
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(q_values, target_value)
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1) ## clamp gradient to range(-1, 1)
        self.optimizer.step()
        self.loss_val = 0
        self.loss_val += loss.item()
        self.loss_history.append(self.loss_val)

class DDQN_PER_4Graph():
    def __init__(self,
                n_actions,
                input_size,
                device,
                timestep, # input time series feature
                batch_size,
                epsilon,
                max_timestep,
                memory_size,
                learning_rate,
                reward_decay,
                is_training,
                epsilon_min=0.01,
                epsilon_decay=0.99):
        self.n_actions = n_actions
        self.input_size = input_size
        self.device = device
        self.timestep = timestep
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.is_training= is_training
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.max_timestep = max_timestep
        self.epsilon_decay = epsilon_decay
        self.diff_eps = (self.epsilon - self.epsilon_min)/self.max_timestep
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.loss_val = 0
        self.loss_history = []
        self.memory = PrioritizedExperienceReplay(memory_size)
        self.policy_net = GAT(timestep, n_actions, batch_size, is_training).to(self.device)
        self.target_net = GAT(timestep, n_actions, batch_size, is_training).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=0.001)
            
    def choose_action(self, observation):
        self.epsilon = self.epsilon - self.diff_eps
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if not self.is_training or np.random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(observation).squeeze().argmax().item()
        else:
            """random action"""
            return random.randint(0, self.n_actions-1)

    def replace_target_net_weight(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self): # Double DQN learn

        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        # batch = []
        # for i in range(len(samples)):
        #     batch.append(PERGraphData(
        #             edge_index_s=samples[i].edge_index_s,
        #             x_s=samples[i].x_s,
        #             edge_index_t=samples[i].edge_index_t,
        #             x_t=samples[i].x_t,
        #             action=samples[i].action,
        #             reward=samples[i].reward,
        #             idx=idxs[i],
        #             is_weight=is_weights[i]))
        batch = Batch.from_data_list(batch)
        # print(type(batch))
        states = Data(x=batch.x_s, edge_index=batch.edge_index_s) #
        next_states = Data(x=batch.x_t, edge_index=batch.edge_index_t)
        actions = batch.action.view(self.batch_size, 1).to(self.device)
        rewards = batch.reward.to(self.device)
        # print(type(states))
        # print(f'states.x.shape:{states.x.shape}')
        # print(f'states.x.shape:{states.edge_index.shape}')
        # print(next_states)
        # print(actions.shape)
        # print(rewards.shape)
        q_values = self.policy_net(states).squeeze()
        # print(f'q_values.shape:{q_values.shape}')
        print(f'q_values:{q_values}')
        q_values = q_values.gather(dim=1, index=actions).squeeze()
        target_actions = self.policy_net(next_states).squeeze().view(-1, self.n_actions).max(1)[1].view(self.batch_size, 1).detach()
        next_q_values = self.target_net(next_states).squeeze().view(-1, self.n_actions).gather(dim=1, index=target_actions).squeeze()
        reward_decay_v = torch.tensor(self.reward_decay, dtype=torch.float).to(self.device)
        target_value = rewards + reward_decay_v * next_q_values

        # update priority
        errors = torch.abs(q_values - target_value).data.cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # Compute loss        
        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weights).to(self.device) * F.mse_loss(q_values, target_value)).mean()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1) ## clamp gradient to range(-1, 1)
        self.optimizer.step()

        self.loss_val = 0
        self.loss_val += loss.item()
        self.loss_history.append(self.loss_val)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    n_actions = 9
    model = DCRQN(n_actions).to(device)
    # x = torch.from_numpy(np.array([0.0000, 0.0889, 0.0000, 0.0000, 0.1037, 0.0307, 0.0047, 0.0000,0.0786]))

    x = torch.rand(32, 1, 64, 9).to(device)
    # print(x)
    # print(output)
    x = model(x)
    # print(x)
    # x = torch.randn([32, 1, 64, 9]).to(device)
    # summary(model,input_size=(1, 64, 9))
    # pass