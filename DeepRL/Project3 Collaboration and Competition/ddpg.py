import random
import torch
import torch.optim as optim
from ActorCriticNets import ActorNetwork, CriticNetwork
from BufferNoise import ReplayBuffer, OUNoise
import numpy as np

# The parameters used for the agent
           # for soft update of target parameters, dfferent from DQN
LR_ACTOR = 5e-4              # learning rate for actor
LR_CRITIC = 1e-3     # learning rate for critic
L2_DECAY_CRITIC = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A modified ddpg agent, sharing a replay buffer across a list of agents
class Agent():
    def __init__(self, state_size, action_size, seed, n_agents=2):
        """Initialize an Agent object.

        Params
        ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_agents = n_agents
        # Actor Networks
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.optim_actor = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        #Critic Networks
        self.critic_local = CriticNetwork(state_size * n_agents, n_agents* action_size, seed).to(device)
        self.critic_target = CriticNetwork(state_size * n_agents, action_size * n_agents, seed).to(device)
        self.optim_critic = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC, weight_decay=L2_DECAY_CRITIC)
       
        # Initialize noise
        self.noise = OUNoise(action_size, seed, sigma=0.2)
        self.t_step = 0
        
  
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Change the network to evaluation mode
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy() + eps * self.noise.sample()
        # Change the network to training mode
        self.actor_local.train()
        return np.clip(action_values, -1, 1)
       
   
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
         θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




