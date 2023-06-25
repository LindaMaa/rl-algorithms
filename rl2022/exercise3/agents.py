from abc import ABC, abstractmethod
from copy import deepcopy
import gym
from importlib_metadata import distribution
import numpy as np
import random
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.exercise3.networks import FCNetwork
from rl2022.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

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

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int,dd):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """DQN agent

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )
        self.MSE=torch.nn.MSELoss()

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = 0.5
        # ######################################### #
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        # best params: max_deduct=0.09, decay=1.31 for cartpole
        max_deduct, decay = 0.09, 1.32
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        n = random.uniform(0,1)
        q_list = self.critics_net(torch.from_numpy(np.array([obs])).float()).detach().numpy()

        # return random function
        if (explore==True and n<self.epsilon):
            actions=q_list.size
            any_action=np.random.choice(actions)
            return any_action
        else:
            # return best action possible
            best_action=np.argmax(q_list)
            return best_action
        
    
    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN


        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        self.update_counter = self.update_counter + 1
        self.critics_optim.zero_grad()

        # parse parameters
        states=batch[0]
        actions=batch[1]
        next_states=batch[2]
        rewards=batch[3]
        done_vals = batch[4]

        # calculating Q vals
        curr_Q = self.critics_net(states)
        curr_Q=curr_Q.gather(1, actions.long())
        next_Q = self.critics_target(next_states)
        max_next_Q= torch.max(next_Q, 1)[0].detach().unsqueeze(1)
        expected_Q = rewards + (1 - done_vals) * self.gamma * max_next_Q
       
        # compute value loss
        q_loss = self.MSE(curr_Q,expected_Q)
        q_loss.backward()
        self.critics_optim.step()

        # update at the given target update frequency
        period_end=(self.update_counter % self.target_update_freq ==0)
        
        if period_end:
            self.critics_target.hard_update(self.critics_net)

        return {"q_loss":  q_loss.detach().numpy()}


class Reinforce(Agent):
    """Reinforce agent

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #
        self.action_space=action_space.n

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self,train_p,timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.learning_rate=train_p

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        # input observation vector to the network, get probabilities of actions
        input_state = torch.from_numpy(np.array(obs)).float()
        probabilities = self.policy.forward(input_state) 
        #distribution= Categorical(logits=probabilities)
        #act = distribution.sample()

        # sample discrete action
        act=np.random.choice(self.n_actions, p=np.squeeze(probabilities.detach().numpy())) 
        return act
        #act=distribution.sample()
        #prob=distribution.log_prob(act)
        #return act.item()
       
    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """

        # initialise loss and returns to 0 for each episode
        p_loss = float(0)
        G = 0

        # loop backwards in the episode
        for step in reversed(range(len(observations))):

            # update returns and p_loss
            G = G * self.gamma  + rewards[step] 

            s = torch.from_numpy(np.array([observations[step]])).float()
            probabilities = self.policy.forward(s)
            distribution = Categorical(probabilities)
            p_loss =  p_loss  - G * distribution.log_prob( torch.from_numpy(np.array([actions[step]])).float()) #log prob of action given state and policy parameters


        # divide by the total number of steps, perform a gradient step
        p_loss = p_loss  / len(observations)
       
       # set gradients to 0 and do backpropagation
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return {"p_loss": float(p_loss)}
