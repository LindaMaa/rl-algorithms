import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.exercise3.agents import Reinforce
from rl2022.exercise3.train_reinforce import CARTPOLE_CONFIG
from rl2022.exercise3.train_reinforce import train

plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})

def plot_timesteps(
        values: np.ndarray,
        eval_freq: int,
        title: str,
        xlabel: str,
        ylabel: str,
        legend_name: str,
    ):
    """
    Plot values with respect to timesteps

    :param values (np.ndarray): numpy array of values to plot as y-values
    :param eval_freq (int): number of training iterations after which an evaluation is done
    :param title (str): name of algorithm
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    plt.figure()
    plt.title(title)
    
    x_values = eval_freq + np.arange(len(values)) * eval_freq

    # plot means with respective standard deviation as shading
    plt.plot(x_values, values, label=f"{legend_name}")

    # set legend and axis-labels
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)


def plot_timesteps_shaded(
        means: np.ndarray,
        stds: np.ndarray,
        eval_freq: int,
        title: str,
        xlabel: str,
        ylabel: str,
        legend_name: str,
    ):
    """
    Plot mean and std-shading for values with respect to timesteps

    :param means (np.ndarray): numpy array of mean values to plot as y-values
    :param stds (np.ndarray): numpy array of standard deviations to plot as y-value shading
    :param eval_freq (int): number of training iterations after which an evaluation is done
    :param title (str): name of algorithm
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    plt.figure()
    plt.title(title)

    x_values = eval_freq + np.arange(len(means)) * eval_freq

    # plot means with respective standard deviation as shading
    plt.plot(x_values, means, label=f"{legend_name}")
    plt.fill_between(
        x_values,
        np.clip(means - stds, 0, 1),
        np.clip(means + stds, 0, 1),
        alpha=0.3,
        antialiased=True,
    )

    # set legend and axis-labels
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)

if __name__ == "__main__":

    CONFIG = CARTPOLE_CONFIG
    RENDER = False
    SEEDS = [i for i in range(10)]
    
    env = gym.make(CONFIG["env"])
    dict_rewards={}
    dict_variance={}
    #l=[0.3,0.2,0.15,0.1,0.05,0.075,0.09,0.08]
    #l=[0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.0065,0.006,0.0055,0.005,0.004,0.003,0.002,0.001]
    #l=[0.0058,0.0059,0.006,0.0061,0.0062,0.0063,0.0057,0.0056]
    #l=[0.1,0.05,0.01,0.005,0.001,0.005,0.0001] 
    #l=[0.0001*0.47,0.0001*0.465,0.0001*0.475,0.0001*0.471,0.0001*0.472,0.0001*0.473,0.0001*0.474,0.0001*0.469,0.0001*0.468,0.0001*0.467,0.0001*0.466]   
    #l=[0.0001] best lr
    l=[5e-05,5.1*1e-05,5.2*1e-05,4.9*1e-05,4.8*1e-05]
    for lr in l:
        lr=lr
        rewards=[]
        learn=(lr)
        for seed in SEEDS:
            for i in range(1):
                    #print(f"Training for seed={seed}")
                random.seed(seed)
                np.random.seed(seed)

                reward,t = train(learn,env, CONFIG,output=False)
                rewards.append(reward[-1:])
                print(reward)
        print("FINAL EVAL")
        print("L val: ", learn)
        dict_rewards[lr]=np.mean(rewards)
        dict_variance[lr]=np.std(rewards)
        print(dict_rewards)
        print(dict_variance)