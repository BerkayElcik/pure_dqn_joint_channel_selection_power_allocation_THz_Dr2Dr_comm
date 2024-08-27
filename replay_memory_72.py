import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=object)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=object)

        self.action_memory = np.zeros(self.mem_size, dtype=object)
        self.action_index_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, action_index, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        """
        print("mmem")
        print(self.mem_size)
        print("state")
        print(state)
        print(state.shape)
        print(state[0].shape)
        print(state[0])
        """
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.action_index_memory[index] = action_index
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        action_indices = self.action_index_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, action_indices, rewards, states_, terminal
