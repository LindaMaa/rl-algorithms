import os
import gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.exercise3.agents import Agent
from rl2022.exercise3.networks import FCNetwork
from rl2022.exercise3.replay import Transition


class DDPG(Agent):
    """DDPG agent

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critic (FCNetwork): fully connected critic network
    :attr critic_optim (torch.optim): PyTorch optimiser for critic network
    :attr policy (FCNetwork): fully connected actor network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for actor network
    :attr gamma (float): discount rate gamma
    """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        # self.actor = Actor(STATE_SIZE, policy_hidden_size, ACTION_SIZE)
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )

        self.actor_target.hard_update(self.actor)
        # self.critic = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)
        # self.critic_target = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)
        self.MSE= torch.nn.MSELoss()

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #
        mu=0
        std=0.1
        self.n_actions=ACTION_SIZE
        self.epsilon=0.1
        
        self.noise = Normal(mu, std * self.n_actions)

        

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )


    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path


    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """

        # observe state and select action produced by the actor network
        best_action = self.actor(torch.from_numpy(np.asarray(obs)).float())
        best_action=np.asarray(best_action.detach())

        if not(explore):
            # greedy action
            return best_action

        # exploration, add noise 
        if explore:
            noisy_action = best_action + np.asarray(self.noise.sample())
            return  np.clip(noisy_action , self.lower_action_bound, self.upper_action_bound)

        
        
            
     
        

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critic and actor networks, target networks with soft
        updates, and return the q_loss and the policy_loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        q_loss = 0
        p_loss = 0

        # parse experience from replay buffer
        states=batch[0]
        actions=batch[1]
        next_states=batch[2]
        rewards=batch[3]
        done_vals = batch[4]
        
        # get original Q values using critic network
        arr_sa=torch.cat((states,actions), dim=-1)
        original_Q = self.critic(arr_sa)

        # get next state Q values using target networks
        arr_next_sa=torch.cat((next_states,self.actor_target(next_states)), dim=-1)
        next_Q = self.critic_target(arr_next_sa)
        
        # use Bellman equation to update Q
        updated_Q = rewards +  self.gamma * next_Q * (1 - done_vals) 

        # calculate critic loss and update critic network
        q_loss = self.MSE(original_Q , updated_Q)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # calculate actor loss and update actor network
        critic_arr=torch.cat((states,self.actor(states)), dim=-1)
        p_loss = (- 1) * self.critic(critic_arr).mean() # mean of sum of gradients
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        # soft update of target networks
        self.actor_target.soft_update(source=self.actor, tau=self.tau)
        self.critic_target.soft_update(source=self.critic, tau=self.tau)

        return {
            "q_loss": q_loss,
            "p_loss": p_loss,
        }
