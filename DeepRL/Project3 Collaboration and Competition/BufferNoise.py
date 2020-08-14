from collections import namedtuple, deque
import numpy as np
import copy
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        #print("len experiences:{}".format(len(experiences)))
        #print("shape experiences state:{}".format(experiences[0].state[0].shape))
        #print("shape experiences action:{}".format(experiences[0].action.shape))
        #print("shape experiences reward:{}".format(experiences[0].reward))
        states = torch.from_numpy(np.vstack([np.dstack(e.state.T) for e in experiences if e is not None])).float().to(device)
        #print("State shape after stack: {}".format(states.shape))
        actions = torch.from_numpy(np.vstack([np.dstack(e.action.T) for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.dstack(e.next_state.T) for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([np.dstack(e.done) for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.standard_normal(len(x))])
        self.state = x + dx
        return self.state
    
    


    