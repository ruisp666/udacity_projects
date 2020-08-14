import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fan_in_dim(layer):
    """
    Returns the fan in dimensions of the layer
    PARAMS
    =====
    layer: The layer of the network
    """
    return layer.weight.data.size()[0]

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.dense1_units = 128
        self.dense2_units = 256
        self.dense3_units = 128
        super(ActorNetwork, self).__init__()
        self.seed = seed
        
        # First layer architecture and initialization scheme
        self.dense1 = nn.Linear(state_size, self.dense1_units)
        r_1 = 1 / np.sqrt(fan_in_dim(self.dense1))
        self.dense1.weight.data.uniform_(-r_1, r_1)

        # Second layer architecture and initialization scheme
        self.dense2 = nn.Linear(self.dense1_units, self.dense2_units)
        r_2 = 1 / np.sqrt(fan_in_dim(self.dense2))
        self.dense2.weight.data.uniform_(-r_2, r_2)
        
        self.dense3 = nn.Linear(self.dense2_units, self.dense3_units)
        r_3 = 1 / np.sqrt(fan_in_dim(self.dense3))
        self.dense3.weight.data.uniform_(-r_3, r_3)

        # Output layer architecture and initialization scheme
        self.output = nn.Linear(self.dense3_units, action_size)
        self.output.weight.data.uniform_(-3e-3, 3e-3)

        # Batch normalization layers
        self.batch_norm_input = nn.BatchNorm1d(state_size, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)
        self.batch_norm_1 = nn.BatchNorm1d(self.dense1_units, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)
        self.batch_norm_2 = nn.BatchNorm1d(self.dense2_units, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)
    
        

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.dense1(state))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        # Use tanh to bound the output between -1 and 1.
        x = torch.tanh(self.output(x))
        #print("output actor: {}".format(x))
        return x
    
class CriticNetwork(nn.Module):
    """ Critic Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
        """
        self.dense1_units = 128
        self.dense2_units = 256
        self.dense3_units = 128
        super(CriticNetwork, self).__init__()
        self.seed = seed
        
        # First layer architecture and initialization scheme
        self.dense1 = nn.Linear(state_size, self.dense1_units)
        r_1 = 1 / np.sqrt(fan_in_dim(self.dense1))
        self.dense1.weight.data.uniform_(-r_1, r_1)
        
        # Second layer architecture and initialization scheme
        # Add the action dimension to the second layer
        self.dense2 = nn.Linear( self.dense1_units + action_size, self.dense2_units)
        r_2 = 1 / np.sqrt(fan_in_dim(self.dense2))
        self.dense2.weight.data.uniform_(-r_2, r_2)
        
        # Third layer architecture and initialization scheme
        self.dense3 = nn.Linear( self.dense2_units, self.dense3_units)
        r_3 = 1 / np.sqrt(fan_in_dim(self.dense3))
        self.dense3.weight.data.uniform_(-r_3, r_3)
       

        # Output layer architecture and initialization scheme
        self.output = nn.Linear(self.dense3_units, 1)
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        

        # Batch Normalization
        self.batch_norm_input = nn.BatchNorm1d(state_size, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)
        self.batch_norm_1= nn.BatchNorm1d(self.dense1_units, eps=1e-05, momentum=0.1,
                                         affine=True, track_running_stats=True)

    def forward(self, state, action):
        """Build a network that maps states, action -> Q(s,a)."""
        #print(state.shape)
        #print(torch.flatten(state, start_dim = 1 ).shape)
        x = F.relu(self.dense1(torch.flatten(state, start_dim=1 )))
        #  Concatenation to include action
        #print("First Layer Shape: {}".format(x.shape))
        #print(action.float().shape)
        z = torch.cat((x, torch.flatten(action.float(), start_dim=1)), dim=1)
        #  Concatenation to include action
        z = F.relu(self.dense2(z))
        z = F.relu(self.dense3(z))
        #print(z.shape)
        z = self.output(z)
        #print("output critic: {}".format(z))


      
        return z
    
