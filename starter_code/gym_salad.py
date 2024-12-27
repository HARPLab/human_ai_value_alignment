# Importing necessary libraries for OpenAI Gym
import gym
from gym import Env, spaces
import numpy as np

# Define a custom environment for Salad-Making MDP
class SaladMakingEnv(Env):
    def __init__(self):
        super(SaladMakingEnv, self).__init__()
        
        # Define action space: Measure, Clean, Cut, Mix, Add Dressing, Serve, Clarify, Substitute
        self.action_space = spaces.Discrete(6)
        
        # Define state space: 10 possible states in the salad-making process
        self.observation_space = spaces.Discrete(8)
        
        # State mapping
        self.states = [
            "Start", "Measuring", "Cleaning", "Cutting", "Mixing", 
            "Dressing", "Serving", "Task Complete"
        ]
        
        # Actions mapping
        self.actions = [
            "Measure", "Clean", "Cut", "Mix", "Add Dressing", 
            "Serve", "Invalid"
        ]
        
        # Initial state
        self.state = 0  # Start state
        
        # Reward mapping
        self.rewards = {
            "Measure": 10, "Clean": 10, "Cut": 10, 
            "Mix": 10, "Add Dressing": 10, "Serve": 100,
            "Invalid": -10
        }
        
        # Transition mapping
        self.transitions = {
            0: {"Measure": 1}, # Start -> Measuring or Clarification
            1: {"Clean": 2},   # Measuring -> Cleaning or Clarification
            2: {"Cut": 3},     # Cleaning -> Cutting or Clarification
            3: {"Mix": 4},     # Cutting -> Mixing or Clarification
            4: {"Add Dressing": 5},  # Mixing -> Dressing or Clarification
            5: {"Serve": 6},   # Dressing -> Serving or Clarification
            6: {"Complete": 7},              # Serving -> Task Complete
        }

    def step(self, action):
        # Get action name
        action_name = self.actions[action]
        
        # Determine the next state
        if action_name in self.transitions[self.state]:
            next_state = self.transitions[self.state][action_name]
        else:
            next_state = self.state  # No state change for invalid actions
            action_name = "Invalid"  # Mark as invalid action
        
        # Calculate reward
        reward = self.rewards.get(action_name, -10)
        
        # Check if the episode is done
        done = self.state == 7  # Task Complete state
        
        # Update the state
        self.state = next_state
        
        # Return step information
        return self.state, reward, done, {}

    def reset(self):
        # Reset to the initial state
        self.state = 0
        return self.state

    def render(self, mode="human"):
        # Display the current state
        print(f"Current State: {self.states[self.state]}")

# Create the environment
env = SaladMakingEnv()

# Test the environment
obs = env.reset()
env.render()

done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
