from abc import ABC, abstractmethod
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

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


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        actions = []
        for agent in range(self.num_agents):
            n = random.uniform(0,1)
            if (n<self.epsilon):
                 # select random action 
                 random_action = random.randint(0, self.n_acts[agent] - 1)
                 actions.append(random_action)
            else:
                # select greedy action
                q_list = [self.q_tables[agent][i] for i in range(self.n_acts[agent])]
                max_q = max(q_list)  
                # if there are multiple max q vals, select randomly from those max q vals
                best_action=random.choice([i for i,  action_value in enumerate(q_list) if action_value == max_q])
                actions.append(best_action)
        return actions
    
        
    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []

        for agent in range(self.num_agents):

            # agent's qtable, action taken, reward and info about terminal state
            qtable = self.q_tables[agent]
            action = actions[agent]
            reward = rewards[agent]
            terminal = dones[agent]

            # find next maximum value for q -> select best possible q
            next_q = max([qtable[a] for a in range(self.n_acts[agent])])
            self.q_tables[agent][action] = qtable[action] + self.learning_rate * ( reward + (self.gamma*(1-terminal)*next_q)-qtable[action])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct



class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**
        :return (List[int]): index of selected action for each agent
        """
        joint_action = []

        for agent in range(self.num_agents):
            n = random.uniform(0,1)
             # select random action 
            if (n<self.epsilon):
                random_action = random.randint(0, self.n_acts[agent] - 1)
                joint_action.append(random_action)
            
            # select best response action
            else:
                # get agent's qtable and opponent
                qtable = self.q_tables[agent]
                opponent= abs(agent-1)

                expected_values = [] # store EV for every action
                
                for action in range(self.n_acts[agent]):
                    expected_value=0

                    for action_opp in range(self.n_acts[opponent]):

                        #make sure to maintain order (first,second)
                        if (agent==0):
                            expected_value = expected_value + (self.models[agent][action_opp]/max(1,sum(self.models[agent].values()))) * qtable[(action, action_opp)]
                        else:
                            expected_value = expected_value + (self.models[agent][action_opp]/max(1,sum(self.models[agent].values()))) * qtable[(action_opp,action)]
                   
                    # actions and their respective expected values - indexed by action
                    expected_values.append(expected_value)
                
                maximum= np.max(expected_values)
                q_list = [i for i, act_ev in enumerate(expected_values) if act_ev == maximum]
                
                # randomly select best action from the list of max actions in case of ties
                best_response=random.choice(q_list)
                joint_action.append(best_response)

        return joint_action


    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []

        for agent in range(self.num_agents):
            
            # parse data from input
            opponent= abs(agent-1)
            action_agent=actions[agent]
            action_opponent=actions[opponent]
            reward=rewards[agent]
            done=dones[agent]
            qtable = self.q_tables[agent]

            # update counts of opponent's actions
            if not action_opponent in self.models[agent]:
                self.models[agent][action_opponent]=1
            else:
                self.models[agent][action_opponent]=self.models[agent][action_opponent]+1

            # find max EV
            evs = []
            for next_a in range(self.n_acts[agent]):
                ev_next_act = 0
                for opp_next_a in range(self.n_acts[opponent]):
                    if (agent==0):
                        ev_next_act =  ev_next_act + (self.models[agent][opp_next_a] / max(1,sum(self.models[agent].values()))) * qtable[(next_a,opp_next_a)]
                    else:
                        ev_next_act =  ev_next_act  + (self.models[agent][opp_next_a] /max(1,sum(self.models[agent].values()))) * qtable[(opp_next_a,next_a)]

                evs.append(ev_next_act)
            max_ev = max(evs) # use this max val when updating qtable

            # update q-vals
            if (agent==0):
                self.q_tables[agent][(action_agent,action_opponent)]= qtable[(action_agent,action_opponent)] + self.learning_rate * (reward + self.gamma * (1-done) * max_ev - qtable[(action_agent,action_opponent)])
                updated_values.append(self.q_tables[agent][(action_agent,action_opponent)])  
            else:
                self.q_tables[agent][(action_opponent,action_agent)]=qtable[(action_opponent,action_agent)] + self.learning_rate * (reward + self.gamma * (1-done) * max_ev - qtable[(action_opponent,action_agent)])
                updated_values.append(self.q_tables[agent][(action_opponent,action_agent)])      
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.learning_rate=0.5
        max_deduct, decay = 0.95, 0.07
        self.epsilon = 1.0 - (min(1.0, timestep / (decay * max_timestep))) * max_deduct
