[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: ../Project1%20Navigation/change_kernel_jupyter_lab.png


# Project 2: Continuous Control

### Introduction

The work here presented, shows how to train a DDPG-based agent to teach a robotic arm to move to target locations.

For this project, we will use the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.


### Environment details
An example of the appearance of the environment is given below.

![Trained Agent][image1]

**Multi/Single agent**: Single agent.

**Rewards**: A reward of +0.1 is provided every step the agent stays at the target.

**State Space**: The state space has 33 dimensions and contains the agent's  position, rotation, velocity, 
and angular velocities of the arm.

**Actions**: 4 dimensional vector, corresponding to torque applicable to two joints. Every entry in the action vector
 should be a number between -1 and 1.


**Success criteria**: The task is episodic, and in order to solve the environment, an agent must get an average score
 of 30 over 100 consecutive episodes.

![Trained Agent][image1]

### Getting Started



1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the environment file in the directory where you are running the python notebook.

3. Create (and activate) a new environment with Python 3.6. Install [anaconda](https://docs.anaconda.com/anaconda/install/) in order to create the python environment needed using conda.

    #### Linux or Mac:
    conda create --name drlnd python=3.6
    
    source activate drlnd
    #### Windows:
    conda create --name drlnd python=3.6 
    
    activate drlnd

4. Clone the udacity deep reinforcment learning repository and navigate to the python/ folder. Then, install several dependencies.

    git clone https://github.com/udacity/deep-reinforcement-learning.git
    
    cd deep-reinforcement-learning/python
    
    pip install .
    
5. (optional) Install jupyterlab

I used jupyterlab, which supports multi-tab editing, and includes a markdown preview function, useful for readme editing. You can install it using the instructions [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

6. Create an IPython kernel for the drlnd environment.
python -m ipykernel install --user --name drlnd --display-name "deep rl (torch)"
Before running code in a notebook, change the kernel to match the "deep rl (torch)" environment by using the drop-down Kernel menu.

![Kernel change][image2]


### Instructions
Although one can quickly get the agent to train by following the steps on `Continuous_Control.ipynb` (use cell 7 to
 train the agent, and cell 26 to make the trained agent play the game), it is important to note that when loading the
  agent (line "from model import ddpgAgent"):
1. The agent depends on the replay buffer class and on the ActorNetwork and CriticNetwork from the network module.
2. The two neural networks architecture are implemented in the ActorCritic.py module, but their optimisation is carried in the 
agent class. The optimized weights are saved in trained_networks.




