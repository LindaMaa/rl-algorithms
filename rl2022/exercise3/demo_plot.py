import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.exercise3.agents import DQN
from rl2022.exercise3.train_dqn import LUNARLANDER_CONFIG,CARTPOLE_CONFIG, play_episode
from rl2022.exercise3.train_dqn import train

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

    CONFIG_testing = {
    "eval_freq": 2000,
    "eval_episodes": 20,
    "learning_rate": 1e-2,
    "hidden_size": (128, 64),
    "target_update_freq": 5000,
    "batch_size": 16,
    "buffer_capacity": int(1e6),
    "plot_loss": False, 
}
    CONFIG = CARTPOLE_CONFIG
    RENDER = False
    SEEDS = [i for i in range(10)]
    
    env = gym.make(CONFIG["env"])
    dict_rewards={}
    #dd 0.8, 1.3
    # max deduct 0.12
    for dd in range(1,21):
        rewards=[]
        dd=dd/100
        for i in range(2):
            for seed in SEEDS:
                #print(f"Training for seed={seed}")
                random.seed(seed)
                np.random.seed(seed)

                reward,t = train(env, CONFIG,max_deduct=dd, output=False)
                rewards.append(reward[-1:])
        print("FINAL EVAL")
        print("DD val: ", dd)
        dict_rewards[dd*100]=np.mean(rewards)
        print(dict_rewards)

    """
        
        eval_reward_means.append(reward_means)
        eval_reward_stds.append(reward_stds)
        eval_epsilons.append(epsilons)

    eval_reward_means = np.array(eval_reward_means).mean(axis=0)
    eval_reward_stds = np.array(eval_reward_stds).mean(axis=0)
    eval_epsilons = np.array(eval_epsilons).mean(axis=0)
    plot_timesteps_shaded(
        eval_reward_means,
        eval_reward_stds,
        CONFIG["eval_freq"],
        "SARSA Evaluation Returns",
        "Timesteps",
        "Mean Evaluation Returns",
        "SARSA",
    )

    plot_timesteps(
        eval_epsilons,
        CONFIG["eval_freq"],
        "SARSA Epsilon Decay",
        "Timesteps",
        "Epsilon",
        "SARSA",
    )

    plt.show()
    """

        

