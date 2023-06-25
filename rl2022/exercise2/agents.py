from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here


        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        n = random.uniform(0,1)
        if (n<self.epsilon):
            # select  random action 
            random_action = random.randint(0, self.n_acts - 1)
            return random_action
        else:
            # select greedy max action
            q_list = [self.q_table[(obs, act)] for act in range(self.n_acts)]
            max_q = max(q_list)
            # in case there are more actions having max q val, select randomly from the list 
            best_action=random.choice([i for i,  q_val in enumerate(q_list) if q_val == max_q])
            return best_action


           

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience


        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """

        # find max q value for next state
        max_q = max([self.q_table[(n_obs, act)] for act in range(self.n_acts)])
        
        # update q table, only consider max q value for next state if current state is not terminal 
        self.q_table[(obs, action)] = self.q_table[(obs, action)] + self.alpha*( reward + (self.gamma*(1-done)*max_q)-self.q_table[(obs, action)])
        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        
        self.epsilon = 1.0-(min(1.0, timestep/(0.09*max_timestep)))*0.99
        self.alpha = 0.10

class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory 
        :param actions (List[int]): list of indices of applied actions in trajectory
        :param rewards (List[float]): list of received rewards during trajectory 
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        G = 0 

        # loop through trajectory backwards
        for step in reversed(range(len(obses))): 

            # update G according to received rewards 
            G = G*self.gamma+rewards[step] 
           
            # get state action pair at a current step in trajectory
            s_a = (obses[step],actions[step])

          
            # do only if state action pair was not seen in previous steps - this means first visit
            if s_a not in list(zip(obses[0:step], actions[0:step])):
                    
                # update sa_counts of state action pair 
                if s_a not in self.sa_counts:
                    self.sa_counts[s_a] = 1 # first occurence
                else:
                    self.sa_counts[s_a] = self.sa_counts[s_a] + 1 

                updated_values[s_a] = (G + self.q_table[s_a] * (self.sa_counts[s_a] - 1)) / self.sa_counts[s_a]

                # update computed average of the returns following first visit of (s,a)
                self.q_table[s_a] =  updated_values[s_a]
        
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0 - (min(1.0, timestep / (0.32 * max_timestep))) * 0.4
    
