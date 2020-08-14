#Multi agent DDPG
from ddpg import Agent
from  BufferNoise  import ReplayBuffer
import numpy as np
import torch

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-2         # for soft update of target parameters, dfferent from single agent ddpg
UPDATE_EVERY = 15 # Frequency at which to update
LEARNING_REPEATS = 5 # Number of learning updates


class MultiAgent():
    def __init__ (self, n_agents, state_size, action_size, seed):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed  
        self.agents = [Agent(self.state_size, self.action_size, self.seed) for i in range(n_agents)]
       
        # Single Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    
        # Gamma
        self.Gamma = GAMMA
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
            # Save experience in replay memory
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            #print("state shape: {}".format(state.shape))
            # reshape state 
            #state = torch.tensor(state).view(-1, self.n_agents * self.state_size)
            #print("state shape: {}".format(state.shape))
            #print("adding to memory")
            self.memory.add(state, action, reward, next_state, done)
            # If enough samples are available in memory, get random subset and learn
            if self.t_step == 0:
                if len(self.memory) > BATCH_SIZE:
                    # Learn 
                    #print("Learning")
                    for i in range(LEARNING_REPEATS):
                        #experiences = self.memory.sample()
                        self.learn(GAMMA)
                        self.update()
                    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            re
        """
        
        # return an array of the actions
        return np.vstack([agent.act(state[i], eps) for i, agent in enumerate(self.agents)])
       
    def learn(self, GAMMA):
        def loss_actor(output, target):
            return (target - output)**2
        
        def policy(states, vector_agent):
            # Take the composite function
            pred_actions = torch.stack([agent.actor_local(next_states[:, j, :]) for j, agent in enumerate(self.agents)], dim=1)
            # Take the derivative with respect to the  vector agent
            f = vector_agent.critic_local(states, pred_actions)
            return f
        experiences = self.memory.sample()   
        for i, agent in enumerate(self.agents):
            #print(experiences)
            states, actions, rewards, next_states, dones = experiences
            #print("experiences for agent {}, : {}".format(i, experiences))
            #print("state after memory sample shape: {}".format(states.shape))
            #print("actions after memory sample shape: {}".format(actions.shape))
            #print(self.agents[1].actor_target(next_states).shape)
            #print(next_states)
            #print("next states after gather: {}".format(next_states[:,i,:]))
            #print(" pred actions shape: {}".format(self.agents[1].actor_target(next_states[:, i, :]).shape))
            # Use the actor to get the action to take (bounded by -1, 1) for each of the agents
            #print("experiences, next state for agent {}: {}".format(i, next_states[:, i, :]))
            #print("Entering pred actions calculation for agent {}".format(i))
            pred_actions = torch.stack([agent.actor_target(next_states[:, j, :]) for j, agent in enumerate(self.agents)], dim=1)
            #print("pred_actions {}:" .format(pred_actions))
            #print("next_states {}".format(next_states))
            
            #print("next_states flattened {}".format(torch.flatten(next_states, start_dim=1 )))
            #print("pred_actions flattened {}".format(torch.flatten(pred_actions, start_dim=1 )))
            # Copy the rewards to pred_critic
            #print(" pred actions for all the agents shape : {}".format(pred_actions.shape))
            #print("rewards: {}".format(rewards))
            pred_critic = rewards[:, i].view(-1,1)
            #print(" pred_critic {}".format(pred_critic))
            #print("pred_critic shape {}".format(pred_critic.shape))
            #print("dones shape {}".format(dones.shape))
            # check where the agent is done
            #print(" dones: {}".format(dones))
            false_idx = dones[:, :, i] == False
            #print(false_idx.shape)
            #print(" rewards shape {}".format(rewards.shape))
            # Apply the update for those states where done=False for the agent
            #print(" next_states for all the agents shape: {}".format(next_states.shape))
            #print(agent.critic_target(next_states, pred_actions).shape)
            #print("false idx :{}".format(false_idx))
            pred_critic[false_idx] += GAMMA * agent.critic_target(next_states, pred_actions)[false_idx] 
                                    #+ rewards[:, i].view(-1, 1)[false_idx]
            #print(pred_critic == rewards[:, i].view(-1, 1))
            #print("actions {}".format(actions))
            
            # Update the critic
            pred_critic_local = agent.critic_local(states, actions)
            L_critic = loss_actor(pred_critic_local, pred_critic).mean()
            agent.optim_critic.zero_grad()
            L_critic.backward(retain_graph=True)
            # Clip the gradient
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.optim_critic.step()
        
            # Update the actor (maximize gradient so taking the symmetric of the loss)
            #print("states before actor:{}".format(states.shape))
            L_actor = - policy(states, agent).mean()
            agent.optim_actor.zero_grad()
            L_actor.backward() 
            agent.optim_actor.step()
            
    def update(self):
         # Update the target networks (when)
        for agent in self.agents:
            agent.soft_update(agent.actor_local, agent.actor_target, TAU)
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)

