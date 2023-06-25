from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

import sys
sys.path.insert(0, '/Users/lindamazanova/Desktop/RLworkspace/uoe-rl2022/')

from rl2022.constants import EX1_CONSTANTS as CONSTANTS
from rl2022.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """
    MDP solver using the Value Iteration algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
  
       # initialize value of all states to 0
        V = np.zeros(self.state_dim)

        while True:
            max_diff = 0 # stopping condition if max_diff<theta
            V_new = np.zeros(self.state_dim)  

            #loop through states and try find max action
            for s in self.mdp.states:
                max_val = 0

                #select max action, summing over all next states and rewards
                for a in self.mdp.actions:
                    val=0
                    for s_next in self.mdp.states:
                        val += self.mdp.P[self.mdp.states.index(s),self.mdp.actions.index(a),self.mdp.states.index(s_next)]* (self.mdp.R[self.mdp.states.index(s),self.mdp.actions.index(a),self.mdp.states.index(s_next)]+self.gamma * V[self.mdp.states.index(s_next)])
                    max_val = max(max_val, val)
                V_new[self.mdp.states.index(s)] = max_val 
                max_diff = max(max_diff, abs(V[self.mdp.states.index(s)] - V_new[self.mdp.states.index(s)]))

            V = V_new
            if max_diff < theta:
                break
        return V
   
        
    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """

        policy = np.zeros([self.state_dim, self.action_dim])
        for state in self.mdp.states:
            max_action_value=0
            best_action=None

            for action in self.mdp.actions:
                for trans in self.mdp.transitions:
                    if (trans[0]==state and trans[1]==action):
                        next_s=self.mdp.states.index(trans[2]) # index of next state after performing action
                        max_action_value=max(V[next_s],max_action_value)
                        if (V[next_s]==max_action_value):
                            best_action=action
                            
            for action in self.mdp.actions:
                if (action==best_action):
                    policy[self.mdp.states.index(state)][self.mdp.actions.index(action)]=1
                else:
                    policy[self.mdp.states.index(state)][self.mdp.actions.index(action)]=0      
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

    
        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)

        while True:
            max_diff = 0
            V_new = np.zeros(self.state_dim) 

            for state in self.mdp.states:
                
                for action in self.mdp.actions:
                    # get probability from policy
                    pi_prob=policy[self.mdp.states.index(state)][self.mdp.actions.index(action)]
                    
                    for next_s in self.mdp.states:
                        V_new[self.mdp.states.index(state)]+= pi_prob * self.mdp.P[self.mdp.states.index(state),self.mdp.actions.index(action),self.mdp.states.index(next_s)] * (self.mdp.R[self.mdp.states.index(state),self.mdp.actions.index(action),self.mdp.states.index(next_s)]+ self.gamma * V[self.mdp.states.index(next_s)]) 
                max_diff = max(max_diff, abs(V[self.mdp.states.index(state)] - V_new[self.mdp.states.index(state)]))
            V = V_new
            if (max_diff<self.theta):
                return np.array(V)
        
    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached


        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        # 1. initialization
        policy = np.zeros([self.state_dim, self.action_dim])
        V = np.zeros([self.state_dim])

        stable=False

        # 2. policy improvement
        while not(stable):
            stable=True
            V=self._policy_eval(policy) # get value function for the latest policy
            
            for s in self.mdp.states:
                max_val=V[self.mdp.states.index(s)]
                best_action=np.argmax(policy[self.mdp.states.index(s)]) # current best action
                
                for a in self.mdp.actions:
                    val=0
                    for s_next in self.mdp.states:
                        val += self.mdp.P[self.mdp.states.index(s),self.mdp.actions.index(a),self.mdp.states.index(s_next)]* (self.mdp.R[self.mdp.states.index(s),self.mdp.actions.index(a),self.mdp.states.index(s_next)]+self.gamma * V[self.mdp.states.index(s_next)])
                    max_val = max(max_val, val)
                    if (val==max_val and best_action!=self.mdp.actions.index(a)):
                        best_action=self.mdp.actions.index(a)
                        V[self.mdp.states.index(s)] = max_val # update value function
                
                for a in self.mdp.actions:
                    if (self.mdp.actions.index(a)==best_action) and policy[self.mdp.states.index(s)][best_action]!=1:
                        stable=False
                        policy[self.mdp.states.index(s)][best_action]=1
                    if (self.mdp.actions.index(a)!=best_action and policy[self.mdp.states.index(s)][self.mdp.actions.index(a)]!=0):
                        policy[self.mdp.states.index(s)][self.mdp.actions.index(a)]=0
                        stable=False
        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function - POLICY")
    print(valuefunc)
