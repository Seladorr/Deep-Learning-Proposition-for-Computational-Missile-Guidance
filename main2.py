from ddpg import *
from env2 import *
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


max_eps = 50
max_stp = 6000
best_score= -1.0

episode_plots = []

initial_state = [5000.0, 20.0, 20.0, 120.0]

agent = Agent(alpha=math.pow(10,-3), beta=math.pow(10,-3), tau=0.1, gamma=0.99, n_actions=1, max_size=500000, batch_size=64)

for episode in range(max_eps):
    target_x = []
    target_y = []
    missile_x = []
    missile_y = []
    distance = []
    env2_instance = env2(initial_state, 250, 400, 400, 0.01)
    missile_x.append(0)
    missile_y.append(0)
    target_x.append(5000.0*math.cos(env2_instance.lamda))
    target_y.append(5000.0*math.sin(env2_instance.lamda))
    distance.append(5000.0)
    
    print('episode:', episode)
    for step in range(max_stp):
        prev_state = [env2_instance.r_dot/env2_instance.r_dot_initial, env2_instance.lamda_dot/env2_instance.lamda_dot_initial]
        
        choose_action = agent.choose_action(np.array([prev_state]))      
        new_state, reward, r, done, miss_x, miss_y, tar_x, tar_y = env2_instance.dynamics(choose_action)
        agent.remember(prev_state, choose_action, reward, new_state, done)
        missile_x.append( missile_x[step]+miss_x)
        missile_y.append( missile_y[step]+miss_y)
        target_x.append( target_x[step]+tar_x)
        target_y.append( target_y[step]+tar_y)

        if(done):       
            print('r:', r)
            # print('distance:', math.sqrt(math.pow(missile_x[step+1]-target_x[step+1],2)+math.pow(missile_y[step+1]-target_y[step+1],2)))
            break

    print('score:', env2_instance.score)     
    agent.learn()
    print(done)
    score = env2_instance.score
    # print('score: ', score)
    if episode == 0:
        torch.save(agent.state_dict(), "agent.params.2")
        best_score = score
        print("saved model weights")

    if score > best_score:
        torch.save(agent.state_dict(), "agent.params.2")
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

    episode_plots.append(plt)
    plt.savefig(f'episode_plots/episode_{episode}.png')

    plt.close()

    agent.random_noise.reset()

for episode_plot in episode_plots:
    episode_plot.show()



        
        