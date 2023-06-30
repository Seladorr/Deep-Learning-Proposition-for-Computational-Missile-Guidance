import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
     def __init__(self, beta, name, chkpt_dir='tmp/ddpg'):
        super().__init__()
        self.flatten = nn.Flatten()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.ReLU()
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

     def forward(self, state, action):
        x = T.cat((state, action),dim = 1)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ActorNetwork(nn.Module):
     def __init__(self, alpha, name, chkpt_dir='tmp/ddpg'):
        super().__init__()
        self.flatten = nn.Flatten()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

     def forward(self, state):
        x= state
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
