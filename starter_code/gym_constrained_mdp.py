# Importing necessary libraries for OpenAI Gym
import gym
from gym import Env, spaces
import numpy as np

# Define a custom environment for Salad-Making MDP
class SaladMakingEnv(Env):
    def __init__(self, constraints):
        super(SaladMakingEnv, self).__init__()
        
        # Define action space: Measure, Clean, Cut, Mix, Add Dressing, Serve
        self.action_space = spaces.Discrete(6)
        
        # Define state space: 8 possible states in the salad-making process
        self.observation_space = spaces.Discrete(8)
        
        # State mapping
        self.states = [
            "Start", "Measuring", "Cleaning", "Cutting", "Mixing", 
            "Dressing", "Serving", "Task Complete"
        ]
        
        # Actions mapping
        self.actions = [
            "Measure", "Clean", "Cut", "Mix", "Add Dressing", 
            "Serve"
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
            0: {"Measure": 1},  # Start -> Measuring
            1: {"Clean": 2},    # Measuring -> Cleaning
            2: {"Cut": 3},      # Cleaning -> Cutting
            3: {"Mix": 4},      # Cutting -> Mixing
            4: {"Add Dressing": 5},  # Mixing -> Dressing
            5: {"Serve": 6},    # Dressing -> Serving
        }

        self.constraints = constraints
        self.violations = {key: 0 for key in constraints}  # Tracking constraint violations

    def step(self, action):
        # Get action name
        action_name = self.actions[action]
        
        # Determine the next state
        if action_name in self.transitions.get(self.state, {}):
            next_state = self.transitions[self.state][action_name]
        else:
            next_state = self.state  # No state change for invalid actions
            action_name = "Invalid"  # Mark as invalid action
        
        # Check constraints
        self.check_constraints(action_name)

        # Calculate reward
        reward = self.rewards.get(action_name, -10)

        # Penalty for violations
        reward -= sum(self.violations.values()) * 5
        
        # Check if the episode is done
        done = self.state == 7  # Task Complete state
        
        # Update the state
        self.state = next_state
        
        # Return step information
        return self.state, reward, done, {"violations": self.violations}

    def reset(self):
        # Reset to the initial state
        self.state = 0
        self.violations = {key: 0 for key in self.constraints}
        return self.state

    def render(self, mode="human"):
        # Display the current state
        print(f"Current State: {self.states[self.state]}")
    
    def check_constraints(self, action_name):
        # Check and track violations for each constraint
        for constraint, condition in self.constraints.items():
            if not condition(action_name):  # If the constraint is violated
                self.violations[constraint] += 1

# Example constraints
def calorie_constraint(action):
    # Example: Add Dressing increases calories beyond a limit
    if action == "Add Dressing":
        return False  # Violates calorie constraint
    return True

constraints = {
    "calories": calorie_constraint
}

# Create the environment
env = SaladMakingEnv(constraints)

# Test the environment
obs = env.reset()
env.render()

done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}, Violations: {info['violations']}")

if done:
    print("Salad-making complete!")
else:
    print("Salad-making incomplete.")
