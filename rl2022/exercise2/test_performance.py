import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.exercise2.agents import MonteCarloAgent
from rl2022.exercise2.train_monte_carlo import CONFIG
from rl2022.exercise2.train_monte_carlo import train

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

    CONFIG = CONFIG
    RENDER = False
    SEEDS = [i for i in range(10)]

    env = gym.make(CONFIG["env"])
    dict_rewards={}
    dict_variance={}
    rewards=[]
    #deduct_list=[0.88, 0.79,0.59, 0.56, 0.58, 0.60, 0.61, 0.64, 0.65, 0.68, 0.75, 0.77, 0.79, 0.85]
    try_list=[i for i in range(30,51)]
    
    # 0.88, 0.79,0.59, 0.56, 0.58, 0.60, 0.61, 0.64, 0.65, 0.68, 0.75, 0.77, 0.79, 0.85
    for i in try_list:
        i=i/100
        rewards=[]

        for seed in SEEDS:
                    
            random.seed(seed)
            np.random.seed(seed)
        
            total_reward, evaluation_return_means, evaluation_negative_returns, q_table  = train(i,env, CONFIG)
            rewards.append(evaluation_return_means[-1:])

        idx=int(i*100)
        dict_rewards[idx]=np.mean(rewards)
        dict_variance[idx]=np.std(rewards)
        print(dict_rewards)
        print(dict_variance)
 
       

    """"
    
    env = gym.make(CONFIG["env"])
    dict_rewards={}
    lst=[7.4,7.6, 8.0, 8.1,8.5]
    for lr in lst:
        rewards=[]
        learn=(lr)*1e-2
        for seed in SEEDS:
            for i in range(3):
                    #print(f"Training for seed={seed}")
                random.seed(seed)
                np.random.seed(seed)

                reward,t = train(env, CONFIG,output=False)
                rewards.append(reward[-1:])
        print("FINAL MEAN EVAL")
        print("L val: ", learn)
        dict_rewards[lr*10]=np.mean(rewards)
        print(dict_rewards)
    """