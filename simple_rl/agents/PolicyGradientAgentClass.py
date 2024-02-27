''' PolicyGradientAgentClass.py: Class for a policy gradient agent.

Note: At present, this agent is not implemented.'''

# Python imports.
import random

# Other imports
from simple_rl.agents.AgentClass import Agent

class PolicyGradientAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions, name=""):
        raise NotImplementedError("Policy Gradient has not yet been implemented.")

    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)
        '''
        pass

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Perform a state of policy gradient.
        '''
        pass
