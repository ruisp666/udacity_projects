[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: Scores.png




# A Deep Q-learning based gameplay agent

### 1. Intro

In this document, we describe with some depth the work that was developed in order to successfully teach a DQN based agent to successfully play a game whose objective is to collect yellow bananas, while avoiding blue ones. For more details on the game, how to install it and play it, or the rul please refer to the Readme. Here, we are concerned with the technical aspects, namely:
- How the agent plays and incorporates the rewards that it gets from the actions taken into its learning.
- The actual learning performance.
    
![Trained agent][image1]

An example of gameplay (source:https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

### 2. Learning Algorithm
The base learning algorithm was the DQN network as it appears in the original [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
The DQN algorithm can be seen as a learning process where training instances are collected and added to a replay buffer from which at certain intervals a batch of those images (representing actions taken and rewards) will be randomly selected and the agent will learn upon. The learning process makes use of two networks: The target network, which is used to calculate the targets, and the local network, which is optimised based on the targets predicted by the target network.
While we will not go into a complete description of the algorithm, we will comment on to features that were tuned to the present task based on those used in the DQN example given in the Udacity DQN lesson of the Deep Reinforcement Learning Nanodegree:
- The Buffer Size was set at 100000 images.
- The discount factor was kept standard at 0.99.
- The Learning frequency (i.e. the parameter "UPDATE_EVERY") was kept at 4.
- The epsilon governing the epsilon-greedy was kept between 1 and 0.01 with a 0.995 decay factor.
- The soft update parameter, used to do a soft copy of the local to the target, was kept at 0.001.

#### 2.1 The Q-value function estimator

In order to estimate the Q-Value function, I used a neural network with 32 hidden layers, with input sizes of 64, 128, and 64 respectively. All the hidden layers made use of relu gates. The output layer took a softmax gate with 4 units, equivalent to the dimension of the action space. Please refer to _model.py_ for implementation details. The number of layers and units used was inspired from the hyperparameters that I chose for a neural network that implemented in the Self Driving Car Nanodegree, for Behavioural cloning.
In the present case, however, instead of convolution layers and filters, I used linear layers and units.
It is important to note that although the agent makes use of two instances of this network, only one is optimised, namely the local network. This was carried by means of an Adam optimiser with a learning rate of 4e-5, in batches of 64, as used in the DQN lesson implementation.
### 3. Performance

The objective of this exercise was to attain an average score of 13 over 100 episodes. As can be seen by the attached notebook, that threshold was attained after 469 episodes. Below, one can see the learning process taking place across episodes.

![Scores][image2]



### 4 Future improvements

An improvement could be to clip the loss error to add stability to the learning process by avoiding explosive gradients, or use prioritised experience replay to make the learning process more efficient by selecting those instances more singificant to learning. 
We could also have used actor critic methods to improve learning performance (i.e. the agent's average score per episode ). This procedure uses one local-target pair of networks to approximate the q-value function, and another similarly structured pair of networks to approximate the policy. The optimization is then carried both on the local q-value network, and on the local policy network.