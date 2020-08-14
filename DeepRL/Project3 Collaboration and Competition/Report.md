[//]: # (Image References)

[image1]: Scores_attempt_at_5500.jpg  'Agent after 5500 episodes'


# A Multi-Agent Deep Learning collaborative approach to play Tennis

### 1. Intro

In this document, we describe he work that was developed in order to successfully teach a
two agents to to play tennis.  Here, we are concerned with the technical aspects, namely:
- How the agents play and use the rewards derived from their actions into learning.
- The actual learning performance and stability.
- Future explorations.

**Note**: Exact reproducibility is not warranted, so one may have to 
restart the training more than one time to actually get effective learning. This shortcoming is a natural consequence of 
the non-deterministic motion of some components of the environments.  Notwithstanding this apparent limitation, the
 learning algorithm is able to generalize well enough so that after a few trials (3-5, with the same parameters) success should
 be expected, alhtough
For more details on the environment or how to install it, please refer to the Readme.


### 2. Learning Algorithm
The underlying reinforcment learning algorithm was the Multi Agent Deep DPG framework, as proposed by Lowe et al. in their
[paper](https://arxiv.org/pdf/1706.02275.pdf). Building on the DDPG [framework](https://arxiv.org/abs/1509.02971), the authors propose a multi-agent approach, where each of the agents uses and independent actor, observing local states, and an independent critic, observing all states, and actions. I implemented this algorithm from scratch. The parameters I used were: 

- The Buffer Size (shared and non-prioritised) was set at 1000000.
- The discount factor was kept standard at 0.99.
- The Learning frequency ("UPDATE_EVERY") was kept at 15.
- The agents learned for 5 times every update.
- The epsilon governing the noise was kept between 1 and 0.01 with a 0.999 decay factor.
- The learning rates for the critic and actor were 0.001 and 0.005 respectively, for each agent.
- The soft update parameter (TAU(, used to do a soft copy of the local to the target, was kept at 0.01, where in the original ddpg is kept at 0.001.

### 3. The Code Structure
The implementation makes use of the following auxiliary modules:

- BufferNoise: It defines and implements both the replay buffer class (to store experience tuples) and the noise process.
The replay buffer class was adapted (modifications on the sample method of ReplayBuufer) from the Continuous Control project. In particular, I modified the sample method of ReplayBuffer, in order to get the right shape for the components in a multi-agent framework.

**Note**: As mentioned in the previous section, in order to make the noise generator Gaussian, I have changed the sampling from uniform to normal. 


- ddpg: It implements the architecture, and noise, act, and soft update methods for a single agent

- ActorCriticNets: It defines the networks of the actor and the critic for each agent.

- multi_agent_ddpg: Implements the Multi Agent DDPG framework.

We refer the reader to the continous control project for more details on single-agent DDPG, and now describe each of the two last modules, namely ActorCriticNets and multi_agent_ddpg. The weight initialization scheme
followed the DDPG paper, and the layers are described in the next sections. Importantly enough, the number of layers, their configuration was kept as close as possible to the single agent DDPG implementation for the Continuous Control project. In fact, I have used the same network architectures as in the continous control project using DDPG. This suggests that the right configuration of parameters is more important than the architecture of the networks themselves. We direct the reader to the Report on the Continuus Control project that you can find also in this repository for more experiment results on DDPG.

#### 3.1 The Q-value function estimator (The critic)

Here, I use neural networks with 3 hidden linear ayers. Note that in the MADDPG framework, the critic uses all the states and all the actions in its input. As such the dimnesions of the input layer and the concatenate layer have to account for this.  
The output layer uses a linear gate on 2 units, one unite per each of the dimensions of the action space.
Please refer to _ActorCriticNets.py_ for implementation details. 
It is important to note that although each agent makes use of two instances of this network, only one is optimized, 
namely the local network. 

### 3.2 The Policy function estimator (The actor)
For the actor, I have used three layers once again, all with relu gates, except the out layer, where I used a tanh gate. Note that in an MADDPG framework the actor still observes only the local state pertaining to the current agent. As such, the architecture from single agent DDPG can be reused.


### 3.3 MultiAgent DDPG 

The multiagent class follows the algorithm in the appendix of the MADDPG paper referred above, and implements the following methods:

- __init__ : Makes use of the action, state size, and the number of agents (2, for the present case).
- act: Stacks in a tensor the individual predicted actions, as per agents policy.
- step: Moves the agent one step, and checks if should learn and update parameters, or simply add experience to the buffer. 
- learn: implements the the algorithm in the appendix of the MADDPG paper.
- update: updates the parameters of all the agents by the soft updates, as implemented in the ddpg module.

### 4. Tuning
Finding the optimal parameters proved to be quite tricky. Not only there are a large number of parameters to tune, but
also a particular combination of parameters may need to be retested in order to get an accurate assessment of the
performance. Interestingly enough, and similar to single agent DDPG, many times **early indicators of performance were not informative of future performance**. As one can see by the python notebook, we were able to arrive to an average score of 0.98, but the performance is unstable and drops after Epsiode 5000.

  
![Agent performance over 5500 episodes][image1]

The parameters used were:
- Noise: Standard deviation of 0.2 , expectation of 0, and theta of 0.15, using a standard normal distribution as noise generator. A further 0.99 decay factor was used in each episode.
- Networks: 3 hidden layers with 128, 256 and 128 units respectively.
- Weight initialization: As per the DDPG paper.
- Batch size: 256, after trying 1024, and 512
- Gamma: 0.99.
- seeds tried: 2345, 23456
- Learning frequency: 5 times every 15 steps.
- Learning rates: 1e-3 for the critic and 5e-4 for the actor.
- Buffer size: 1000000.
- The first 20 episodes were used to collect samples based on random actions.


## 5. Future work

A possible improvement could be made by the use of policy ensembles, where in each iteration, we select at random a sub-policy from a set of polices that are trained for each agent. We could have also used
prioritized experience replay to add stability and make the learning process more efficient by selecting those instances
 more significant to learning, or make the architectures of the critic and actor different. We could have tried
 Trust Region Policy Optimization as the base method, which, by constraining the 
 distribution of policy with the help of an additional parameter, was shown to offer improvements in learning stability,
 resulting in better performance, when compared with DDPG, in a single agent framework. 
