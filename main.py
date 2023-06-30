from ddpg import *
from env import *
import math
import numpy as np


max_eps = 10
max_stp = 5000

initial_state = [5000.0, 20.0, 20.0, 120.0]

agent = Agent(alpha=math.pow(10,-3), beta=math.pow(10,-3), tau=0.1, gamma=0.99, n_actions=1, max_size=500000, batch_size=64)

for episode in range(max_eps):
  
    env_instance = env(initial_state, 250, 400, 400, 0.01)
    print('episode:', episode)
    for step in range(max_stp):
        prev_state = [env_instance.r_dot/env_instance.r_dot_initial, env_instance.lamda_dot/env_instance.lamda_dot_initial]
        
        choose_action = agent.choose_action(np.array([prev_state]))      
        new_state, reward, r, done = env_instance.dynamics(choose_action)
        agent.remember(prev_state, choose_action, reward, new_state, done)

        if(done):
            print(new_state)
            break
        
    agent.learn()
    print(done)
    print('score', env_instance.score)
    agent.random_noise.reset()

        
        