import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import *
from buffer import *
        

class Agent(object):
     def __init__(self, alpha, beta, tau = 0.1, gamma = 0.99, n_actions = 1, max_size = 500000, batch_size = 64):
          self.gamma = gamma
          self.tau = tau
          self.memory = ReplayBuffer(max_size)
          self.batch_size = batch_size

          self.actor = ActorNetwork(alpha, name='Actor')
          self.target_actor = ActorNetwork(alpha, name='TargetActor')
          self.critic = CriticNetwork(beta, name='Critic')
          self.target_critic = CriticNetwork(beta, name='TargetCritic')
          self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)
          self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
          self.target_critic_optimizer = optim.Adam(self.target_critic.parameters(), lr=beta)
          self.target_actor_optimizer = optim.Adam(self.target_actor.parameters(), lr=alpha)
          
          sigma_value = 0.1
          theta_value = 0.15
          self.random_noise = noise(mu=np.zeros(n_actions), sigma=sigma_value, theta=theta_value)

          # self.update_network_parameters(tau=0.1)

     def choose_action(self, observation):
          self.actor.eval()
          observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
          
          mu = self.actor(observation).to(self.actor.device)
          action_noise = T.tensor(self.random_noise.generate_noise(), dtype=T.float).to(self.actor.device)
          mu_prime = mu + action_noise
          
          self.actor.train()
          return mu_prime.cpu().detach().numpy()
     
     def remember(self, state, action, reward, new_state, done):
          self.memory.store_transition(state, action, reward, new_state, done)

     def learn(self):  
          if self.memory.mem_cntr < self.batch_size:
               return
          state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
          
          rewards = T.tensor(reward, dtype=T.float).to(self.critic.device)
          new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
          action = T.tensor(action, dtype=T.float).to(self.critic.device)
          state = T.tensor(state, dtype=T.float).to(self.critic.device)

          self.target_actor.eval()
          self.target_critic.eval()
          self.critic.eval()

          target_actions = self.target_actor.forward(new_state)
          critic_value_ = self.target_critic.forward(new_state, target_actions)
          critic_value = self.critic.forward(state, action)

          target = []
          for j in range(self.batch_size):
               target.append(rewards[j] + self.gamma*critic_value_[j]*(done[j]))
          target = T.tensor(target).to(self.critic.device)
          target = target.view(self.batch_size, 1)

          self.critic.train()
          
          self.critic_optimizer.zero_grad()
          critic_loss = F.mse_loss(target, critic_value)
          l2_normc = sum(p.pow(2.0).sum() for p in self.critic.parameters())
          critic_loss = critic_loss + 0.005*l2_normc
          critic_loss.backward()
          self.critic_optimizer.step()

          self.critic.eval()
          
          self.actor.train()
          
          self.actor_optimizer.zero_grad()
          mu = self.actor.forward(state)
          actor_loss = -self.critic.forward(state, mu)
          actor_loss = T.mean(actor_loss)
          l2_norma = sum(p.pow(2.0).sum() for p in self.actor.parameters())  
          actor_loss = actor_loss + 0.006*l2_norma
          actor_loss.backward()
          self.actor_optimizer.step()

          self.update_network_parameters()

     def update_network_parameters(self, tau=0.1 ):
          if tau is None:
               tau = self.tau
          
          actor_params = self.actor.named_parameters()
          critic_params = self.critic.named_parameters()
          target_actor_params = self.target_actor.named_parameters()
          target_critics_params = self.target_critic.named_parameters()

          critic_state_dict = dict(critic_params)
          actor_state_dict = dict(actor_params)
          target_critic_dict = dict(target_critics_params)
          target_actor_dict = dict(target_actor_params)

          for name in critic_state_dict:
               critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()

          self.target_critic.load_state_dict(critic_state_dict)

          for name in actor_state_dict:
               actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()

          self.target_actor.load_state_dict(actor_state_dict)


          
          
