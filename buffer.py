import numpy as np

class noise(object):
    def __init__(self, mu=0, sigma=0.1, theta=0.15, dt=1e-5, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0 
        self.reset()

    def generate_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else 0  

class ReplayBuffer(object):
        def __init__(self, max_size = 500000, input_shape = [2], n_actions = 1):
            self.mem_size = max_size
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size, *input_shape))
            self.new_state_memory = np.zeros((self.mem_size, *input_shape))
            self.action_memory = np.zeros((self.mem_size, n_actions))
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        def store_transition(self, state, action, reward, state_, done):
             index = self.mem_cntr % self.mem_size
             #print(index)
             self.state_memory[index] = state
             self.action_memory[index] = action
             self.reward_memory[index] = reward
             self.new_state_memory[index] = state_
             self.terminal_memory[index] = 1-done
             self.mem_cntr += 1

        def sample_buffer(self, batch_size):
             max_mem = min(self.mem_cntr, self.mem_size)
             batch = np.random.choice(max_mem, batch_size)

             states = self.state_memory[batch]
             new_states = self.new_state_memory[batch]
             rewards = self.reward_memory[batch]
             actions = self.action_memory[batch]
             terminal = self.terminal_memory[batch]

             return states,actions,rewards,new_states,terminal