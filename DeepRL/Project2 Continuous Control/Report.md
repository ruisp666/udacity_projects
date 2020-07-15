[//]: # (Image References)

[image1]: agent_score_17.mp4  'Agent with average score of 17'
[image2]: 
[image3]:
[image4]:




# A generalized Deep Deterministic Policy Gradient for tracking movable objects

### 1. Intro

In this document, we describe he work that was developed in order to successfully teach a
an agent to successfully track the movement of a moving target.  Here, we are concerned with the technical aspects, namely:
- How the agent plays and incorporates the rewards that it gets from the actions taken into its learning.
- The actual learning performance and stability.
- Future explorations.

**Note**: Exactly reproducibility is not warranted, so one may have to 
restart the training more than one time to actually get effective learning. This shortcoming is a natural consequence of 
the non-deterministic distribution of the moving target over the episodes.  Notwithstanding this apparent limitation, the
 learning algorithm is able to generalize well enough so that after a few trials success should
 be expected.
For more details on the environment or  how to install it, please refer to the Readme.


    
![Trained agent][image1]
An example of the environment (source:https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

### 2. Learning Algorithm
The underlying learning algorithm was the Deep DPG framework, as proposed by Lillicrap, Hunt et al at
(https://arxiv.org/pdf/1509.02971.pdf) [paper]. DDPG can be seen as an actor-critic method, i.e. a framework which use
one estimator for the policy, governing the action of the agent, and another estimator for the Q-value function, i.e.
the function which measures the expected reward if the agent takes a particular action given a certain environment state.

Similarly to the Deep Q-Network, DDPG makes use of experience replay, but adds a noise component, in the form of a 
Orsntein-Ulhenbeck process, which can be seen as the velocity process of a brownian motion.
- The Buffer Size was set at 100000 images.
- The discount factor was kept standard at 0.99.
- The Learning frequency (i.e. the parameter "UPDATE_EVERY") was kept at 4.
- The epsilon governing the epsilon-greedy was kept between 1 and 0.01 with a 0.995 decay factor.
- The soft update parameter, used to do a soft copy of the local to the target, was kept at 0.001.

### 3. The implementation
The implementation makes use of the following auxiliary modules:
- BufferNoise: It defines and implements both the replay buffer class (to store experience tuples) and the noise process.
The replay buffer class was taken from the Navigation project, and the noise process from the classroom's example.
**Note**: In order to make the noise process Gaussian, I have change the sampling from uniform to normal 
(line 72 of BufferNoise.py) 

- ActorCriticNets: It defines the networks of the actor and the critic.

#### 2.1 The Q-value function estimator (The critic)

In order to estimate the Q-Value function, I used a neural network with 3 hidden layers, with input sizes of 64, 128, and 64 respectively. All the hidden layers made use of relu gates. The output layer took a softmax gate with 4 units, equivalent to the dimension of the action space. Please refer to _model.py_ for implementation details. The number of layers and units used was inspired from the hyperparameters that I chose for a neural network that implemented in the Self Driving Car Nanodegree, for Behavioural cloning.
In the present case, however, instead of convolution layers and filters, I used linear layers and units.
It is important to note that although the agent makes use of two instances of this network, only one is optimised, namely the local network. This was carried by means of an Adam optimiser with a learning rate of 4e-5, in batches of 64, as used in the DQN lesson implementation.

###2.2 The Policy function estimator (The actor)




### 3. The implementation


### 4. Performance

The objective of this exercise was to attain an average score of 13 over 100 episodes. As can be seen by the attached notebook, that threshold was attained after 469 episodes. Below, one can see the learning process taking place across episodes.

![Scores][image2]



### 5. Future improvements

An improvement could be to clip the loss error to add stability to the learning process by avoiding explosive gradients, or use prioritised experience replay to make the learning process more efficient by selecting those instances more singificant to learning. 
We could also have used actor critic methods to improve learning performance (i.e. the agent's average score per episode ). This procedure uses one local-target pair of networks to approximate the q-value function, and another similarly structured pair of networks to approximate the policy. The optimization is then carried both on the local q-value network, and on the local policy network.