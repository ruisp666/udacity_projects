[//]: # (Image References)

[image1]: img/agent_score_17.gif   'Agent with average score of 17'
[image2]: img/Scores_attempt_1.jpg
[image3]: img/Scores_attempt_2.jpg
[image4]: img/Scores_attempt_10_at_300.jpg
[image5]: img/Scores_attempt_14_at_500.jpg
[image6]: img/Scores_attempt_14.2_at_500.jpg
[image7]: img/Scores_attempt_16_at_900.jpg 
[image8]: img/Scores_attempt_18_at_375.jpg
[image9]: img/Scores_attempt_25_at_300.jpg
[image10]: img/agent_at_score_30.gif


# A generalized Deep Deterministic Policy Gradient for tracking movable objects

### 1. Intro

In this document, we describe he work that was developed in order to successfully teach a
an agent to successfully track the movement of a moving target.  Here, we are concerned with the technical aspects, namely:
- How the agent plays and incorporates the rewards that it gets from the actions taken into its learning.
- The actual learning performance and stability.
- Future explorations.

**Note**: Exact reproducibility is not warranted, so one may have to 
restart the training more than one time to actually get effective learning. This shortcoming is a natural consequence of 
the non-deterministic motion of the moving target over the episodes.  Notwithstanding this apparent limitation, the
 learning algorithm is able to generalize well enough so that after a few trials success should
 be expected.
For more details on the environment or  how to install it, please refer to the Readme.

    
![Trained agent][image1]

An example of the training environment, with the agent averaging a score of 17.5.

### 2. Learning Algorithm
The underlying learning algorithm was the Deep DPG framework, as proposed by Lillicrap, Hunt et al. in their original
[paper](https://arxiv.org/pdf/1509.02971.pdf). DDPG can be seen as an actor-critic method, i.e. a framework employs 
one estimator for the policy, governing the action of the agent, and another estimator for the Q-value function, i.e.
the function which measures the expected reward if the agent takes a particular action given a certain environment state.

Similarly to the Deep Q-Network, DDPG makes use of experience replay, but adds a noise component, in the form of a 
Orsntein-Ulhenbeck process, which can be seen as the velocity process of a Brownian Motion.
- The Buffer Size was set at 100000 images.
- The discount factor was kept standard at 0.99.
- The Learning frequency (i.e. the parameter "UPDATE_EVERY") was kept at 4.
- The epsilon governing the epsilon-greedy was kept between 1 and 0.01 with a 0.995 decay factor.
- The soft update parameter, used to do a soft copy of the local to the target, was kept at 0.001.

### 3. The implementation
The implementation makes use of the following auxiliary modules:
- BufferNoise: It defines and implements both the replay buffer class (to store experience tuples) and the noise process.
The replay buffer class was taken from the Navigation project, and the noise process from the classroom's example.
**Note**: In order to make the noise process Gaussian, I have changed the sampling from uniform to normal 
(line 72 of BufferNoise.py) 

- ActorCriticNets: It defines the networks of the actor and the critic.

- ddpg_gen_agent: It defines and implements the generalized DDPG agent.

We now describe each of the two last modules, namely ActorCriticNets and ddpg_gen_agent. The weight initialization scheme
followed the DDPG paper.

#### 3.1 The Q-value function estimator (The critic)

Here, I experimented with  neural networks with 3 hidden linear ayers. All the hidden layers make use of Relu gates.
The output layer uses of a linear gate on 4 units, one unite per each of the dimensions of the action space.
Please refer to _ActorCriticNets.py_ for implementation details. 
It is important to note that although the agent makes use of two instances of this network, only one is optimized, 
namely the local network. 

### 3.2 The Policy function estimator (The actor)
For the actor, we experimented with a neural network with 3 and 2 hidden linear layers. However, in order to include the
action vector, we have concatenated the first layer with the action input before applying the second layer. All the hidden
layers made use of Relu gates, while the output layer was gated by the hyperbolic tangent in order to bound the actions
between -1 and 1.

### 3.3 The Generalized DDPG agent

The agent was based on my implementation of the DQN agent, with the following additions:
- As can be seen in the __init__ method, the agent includes now 4 networks, a (local, target) pair for the actor and for the critic.
- Only the local networks are optimized, both with ADAM, as per the authors suggestion.
- The agent supports non-continuous learning.
- The learning algorithm ('learn' method) was directly implemented from the DDPG paper from scratch.
- The norm of the gradient of the critic was clipped.

### 4. Tuning
Finding the optimal parameters proved to be quite tricky. Not only there are a large number of parameters to tune, but
also a particular combination of parameters may need to be retested in order to get an accurate assessment of the
performance. Interestingly enough, many times **early indicators of performance were not informative of future performance**.

In order to reduce the complexity of the problem, I have decided to use equal number of layers and units for both Actor
 and Critic networks. As such, my first attempt was to use the same learning rates as the paper, and use 128 and 256 
 units for first and second hidden layers of both networks. Finally, I let the agent
  learn at every step, one update. The results, for 600 episodes, can be seen below.

![Scores for attempt 1][image2]

We can see that with an average score of around 2, we were clearly very far from the objective of getting a score of 30!
From the chart, the agent behaviour is clearly too erratic, and at the same time always very far from a score above 3.
As such, for my second attempt, I decided to add on layer to the network in order to improve learning performance, and
 in order to reduce noise, update the agent one time, every 10 steps. All the remaining parameters were kept unchanged.


![Scores for attempt 2][image3]

Interestingly enough, the agent was able to reach scores around 20, which indicated that adding a layer was probably
beneficial. At the same time, the performance of the agent was relatively stable during the first 200-250 episodes, but
then became highly erratic. Overall, the average score was still around 4.5 after 600 episodes. 

After several trials, I have managed to achieve a decent bump in performance at my 10th attempt (details on __trials_log.csv__).
I have also started charting the performance of the agent every 100 episodes. 

![Scores for attempt 10][image4]

By playing around with the number of updates (5 updates every 20 steps), learning rates (refer to the log), we have managed to
reach an average score in the vicinity of 15, which put us at halfway our objective. This time, I also trained the agent
for 1000 episodes, split into two sessions.

![Scores for attempt 14.1][image5]

![Scores for attempt 14.2][image6]


 There is a slight decrease in performance after the average arrives to ~28. 
It is likely that the complexity of the networks is sufficient, since the model is capable of attaining scores above 30.
However, there are several occurrences of low scores (5-10) which are pushing the average down.
 In order to increase the stability of learning, I decided, at my 16th attempt, to reduce the update frequency to 5 updates
  every 30 steps...
 
![Scores for attempt 16][image7]

The maximum average score jumped to 19.2 at episode 900.

### 4.1 Reaching 30
 
Given the results of attempt 16, I have decided to increase the learning rate of the actor to 5e-4 and augment the buffer
size to 500k. Finally, we arrived at a 100-episode average of 30.07 after 375 episodes!

 
  
![Scores for attempt 18][image8]

The learning progress was fast enough that we can see the episode score sits quite frequently above the average
, which is in technical parlance terms, a very bullish signal. To recap, the parameters used were:
- Noise: Standard deviation and expectation the same as the DDPG paper.
- Networks: 3 hidden layers with 128, 256 and 128 units respectively.
- Weight initialization: As per the DDPG paper.
- Batch size: 128.
- Gamma: 0.99.
- Learning frequency: 5 times every 30 steps.
- Learning rates: 1e-3 for the critic and 5e-4 for the actor.
- Buffer size: 500000.

### 4.2 Reaching 31

For my final experiment,  I decided to find the maximum  average score over a particular run,
 using the same parameters as in 3.1. I split the experiment in half in cells 8 and 9 of the notebook, and below you can 
 find the scores from episodes 700 to 1000.

![Scores for attempt 25][image9]

 The maximum achieved was 31.0, at episode 902. The network parameters are saved as checkpoint_critic_max and checkpoint_actor_max. 
 A gif taken when the score was averaging 30.5 follows.
 
![Scoring above 30][image10]

### 5. Future work

An improvement could be to tune the use of batch normalization to achieve better performance.  We could have also used
prioritized experience replay to add stability and make the learning process more efficient by selecting those instances
 more significant to learning, or make the architectures of the critic and actor different. Finally, we could have used
 Trust Region Policy Optimization, which, by constraining the 
 distribution of policy with the help of an additional parameter, was shown to offer improvements in learning stability,
 resulting in better performance, when compared with DDPG.
