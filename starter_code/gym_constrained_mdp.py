# Importing necessary libraries for OpenAI Gym
import gym
from gym import Env, spaces
import numpy as np

class Ingredient:
    def __init__(self, name, quantity, dietary_restrictions=None):
        self.name = name
        self.quantity = quantity
        self.state = "raw"
        self.dietary_restrictions = dietary_restrictions or []
        self.possible_actions = {
            "Measure": ["Measure"],
            "Combine": ["Combine"]
        }
        self.current_step = 0  # Tracks progress through preparation steps

    def update_state(self, action):
        # Check if the action is valid for the current step
        current_action_key = list(self.possible_actions.keys())[self.current_step]
        if action in self.possible_actions[current_action_key]:
            self.state = action.lower()
            self.current_step += 1
        else:
            raise ValueError(f"Invalid action: {action} not allowed at this step.")

    def is_ready(self):
        # Returns True if all steps are completed
        return self.current_step >= len(self.possible_actions)

class Vegetable(Ingredient):
    def __init__(self, name, quantity, dietary_restrictions=None):
        super().__init__(name, quantity, dietary_restrictions)
        self.possible_actions.update({
            "Wash": ["Wash"],
            "Prepare": ["Chop", "Dice", "Shred"]
        })

class Dressing(Ingredient):
    def __init__(self, name, quantity, dietary_restrictions=None):
        super().__init__(name, quantity, dietary_restrictions)
        self.possible_actions.update({
            "Pour": ["Pour"]
        })

class Nuts(Ingredient):
    def __init__(self, name, quantity, dietary_restrictions=None):
        super().__init__(name, quantity, dietary_restrictions)
        self.possible_actions.update({
            "Prepare": ["Crush", "Dice", "Grind"],
            "Roast": ["Roast"]
        })

