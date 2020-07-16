import random
import torch
import torch.optim as optim
from ActorCriticNets import ActorNetwork, CriticNetwork
from BufferNoise import ReplayBuffer, OUNoise
import numpy as np

# The parameters used for the agent
BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 128      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3           # for soft update of target parameters, dfferent from DQN
LR_ACTOR = 5e-4              # learning rate for actor
LR_CRITIC = 1e-3     # learning rate for critic
L2_DECAY_CRITIC = 0
UPDATE_EVERY = 30   # Frequency at which to update
LEARNING_REPEATS = 5 # Numer of learning updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    def __init__(self, state_size, action_size, seed):
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

        # Actor Networks
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.optim_actor = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        #Critic Networks
        self.critic_local = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        self.optim_critic = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC, weight_decay=L2_DECAY_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
       
        # Initialize noise
        self.noise = OUNoise(action_size, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.memory.add(state, action, reward, next_state, done)
        # If enough samples are available in memory, get random subset and learn
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                # Learn 
                for i in range(LEARNING_REPEATS):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
    
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
       
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        def loss_actor(output, target):
            return (target - output)**2
        
        def policy(states):
            # Take the composite function
            f = self.critic_local(states, self.actor_local(states))
            return f
            
        states, actions, rewards, next_states, dones = experiences
        
        # Use the actor to get the action to take (bounded by -1, 1)
        pred_actions = self.actor_target(next_states) 
        # Copy the rewards to pred_critic
        pred_critic = rewards
        false_idx = dones == False
        # Apply the update for those states where done=False
        pred_critic[false_idx] = GAMMA * self.critic_target(next_states, pred_actions)[false_idx] +  rewards[false_idx]
        
        # Update the critic
        pred_critic_local = self.critic_local(states, actions)
        L_critic = loss_actor(pred_critic_local, pred_critic).mean()
        self.optim_critic.zero_grad()
        L_critic.backward()
        # Clip the gradient
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.optim_critic.step()
        
        # Update the actor (maximize gradient so taking the symmetric of the loss)
        L_actor = - policy(states).mean()
        self.optim_actor.zero_grad()
        L_actor.backward() 
        self.optim_actor.step()
        
        # Update the target networks
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
        
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




