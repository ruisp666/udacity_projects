[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: change_kernel_jupyter_lab.png "Change Kernel"

# Project 1: Navigation

### Introduction

The work here displayed, shows how to train a DQN-based agent to play a game in a large, square world.  The aim is to collect as many yellow bananas as possible, while avoiding the blue bananas.



### Environment details
An example of the appearance of the environment is given below.

![Trained Agent][image1]

**Multi/Single agent**: Single agent.

**Rewards**: A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  

**State Space**: The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  

**Actions**:Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

   They are represented by a vector of size 4.

**Success criteria**: The task is episodic, and in order to solve the environment, an agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the directory where you are running the python notebook.

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

Although one can quickly get the agent to train by following the steps on `Navigation.ipynb` (use cell 7 to train the agent, and cell 26 to make the trained agent play the gam), it is important to note that when loading the agent (line "from model import Agent"):
1. The agent depends on the replay buffer class and on the Qnetwork from the network module.
2. The neural network architecture is implemented in the Qnetwork class, but its optimisation is carried in the agent class. The optimized weights are saved in __working_model.pth__.



