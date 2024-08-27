import numpy as np
import torch as T
from deep_q_network_72 import DeepQNetwork
from replay_memory_72 import ReplayBuffer
from math import sqrt
from utils import cartesian

class DQNAgent(object):
    def __init__(self, n_actions, input_dims, gamma=0.99, epsilon=1, lr=0.001,
                 mem_size=2000, batch_size=256, eps_min= 0.01, eps_dec=5e-7,
                 replace=100, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.mem_size=mem_size

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_prime = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_prime',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        #self.epsilon=0 #for debugging
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action_index=T.argmax(actions).item()
            action_index_array=np.arange(sqrt(self.n_actions))
            action_index_arrays=[action_index_array, action_index_array]
            action_mat=cartesian(action_index_arrays)
            action=action_mat[action_index]
            action=action.astype(np.int32)



        else:
            action_index=np.random.randint(0,self.n_actions-1)
            action_index_array = np.arange(sqrt(self.n_actions))
            action_index_arrays = [action_index_array, action_index_array]
            action_mat = cartesian(action_index_arrays)
            action = action_mat[action_index]
            action = action.astype(np.int32)

        return action, action_index

    def store_transition(self, state, action, action_index, reward, state_, done):
        self.memory.store_transition(state, action, action_index, reward, state_, done)

    def sample_memory(self):
        state, action, action_index, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)


        state = state.astype(np.float32)
        action = np.vstack(action).astype(np.int32)
        new_state=new_state.astype(np.float32)



        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        action_indices = T.tensor(action_index).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)


        return states, actions, action_indices, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_prime.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    """
    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
    """

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_prime.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_prime.load_checkpoint()

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        """
        if self.memory.mem_cntr < self.mem_size:
            return
        """

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, action_indices, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)



        """
        #index of the action  is 
        
        n_channels*actions[0]+actions[1]

        or
        
        #sqrt(n_actions)*actions[0]+actions[1]
        """

        q_pred = self.q_eval.forward(states)[indices, action_indices.long()]
        q_prime = self.q_prime.forward(states_).max(dim=1)[0]

        q_prime[dones] = 0.0
        q_target = rewards + self.gamma*q_prime

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
