import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores_, n, ep):
    """
    Plots a vector of scores and the 100-average
    PARAMS
    =====
    scores_: a List 
    n: The attempt
    ep: The right limit of the 100 epsiode interval
    """
    scores_mean = [np.nan]* len(scores_)
    scores_mean[99:] = [np.mean(scores_[i-100: i]) for i in range(100, len(scores_))]
    plt.plot(np.arange(len(scores_)), scores_, label='episodic score')
    plt.plot(np.arange(len(scores_mean)), scores_mean, label='average across 100 episodes')
    plt.legend()
    plt.title(f'Attempt {n}')
    plt.ylabel('Scores')
    plt.xlabel('Episode #')
    plt.savefig(f'Scores_attempt_{n}_at_{ep}.jpg')
    plt.show()