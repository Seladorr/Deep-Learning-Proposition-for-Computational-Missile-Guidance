from ddpg import *
from env import *
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


max_eps = 10
max_stp = 5000
best_score= -1.0

initial_state = [5000.0, 20.0, 20.0, 120.0]

agent = Agent(alpha=math.pow(10,-3), beta=math.pow(10,-3), tau=0.1, gamma=0.99, n_actions=1, max_size=500000, batch_size=64)

for episode in range(max_eps):
    target_x = []
    target_y = []
    missile_x = []
    missile_y = []
    distance = []
    env_instance = env(initial_state, 250, 400, 400, 0.01)
    missile_x.append(0)
    missile_y.append(0)
    target_x.append(5000.0*math.cos(env_instance.lamda))
    target_y.append(5000.0*math.sin(env_instance.lamda))
    distance.append(5000.0)
    
    print('episode:', episode)
    for step in range(max_stp):
        prev_state = [env_instance.r_dot/env_instance.r_dot_initial, env_instance.lamda_dot/env_instance.lamda_dot_initial]
        
        choose_action = agent.choose_action(np.array([prev_state]))      
        new_state, reward, r, done, miss_x, miss_y, tar_x, tar_y = env_instance.dynamics(choose_action)
        agent.remember(prev_state, choose_action, reward, new_state, done)
        missile_x.append( missile_x[step]+miss_x)
        missile_y.append( missile_y[step]+miss_y)
        target_x.append( target_x[step]+tar_x)
        target_y.append( target_y[step]+tar_y)

        if(done):
            print('r:', r,'score:', env_instance.score)
            # print('distance:', math.sqrt(math.pow(missile_x[step+1]-target_x[step+1],2)+math.pow(missile_y[step+1]-target_y[step+1],2)))
            break
        
    agent.learn()
    print(done)
    score = env_instance.score
    # print('score: ', score)
    if episode == 0:
        torch.save(agent.state_dict(), "agent.params")
        best_score = score
        print("saved model weights")

    if score > best_score:
        torch.save(agent.state_dict(), "agent.params")
        best_score = score
        print("saved model weights")

    plt.plot(missile_x, missile_y, 'r', target_x, target_y, 'b')
    missile_start_x, missile_start_y = missile_x[0], missile_y[0]
    plt.plot(missile_start_x, missile_start_y, marker='*', color='red', markersize=5)
    target_start_x, target_start_y = target_x[0], target_y[0]
    plt.plot(target_start_x, target_start_y, marker='o', color='green', markersize=5)

    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Missile and Target Path')
    plt.legend(['Missile', 'Target', 'Missile Start', 'Target Start'])

    # plt.plot(distance, 'r')
    plt.show()
    
    agent.random_noise.reset()



        
        
