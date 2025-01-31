from gym import Env
from gym.spaces import Discrete, Dict
import numpy as np


class SaladEnv(Env):
    def __init__(self, recipe, constraints):
        super(SaladEnv, self).__init__()

        # Recipe and Constraints are user defined
        self.recipe = recipe 
        self.constraints = constraints 

        self.violations = {"calories": 0, "allergies": [], "availability":{}}

        # Actions we can take: Measure, Wash, Chop, Dice, Shred, Crush, Grind, Roast, Combine
        self.action_map = {
            0: "Meaure", 1: "Wash", 2: "Chop", 3: "Dice", 4: "Shred",
            5: "Crush", 6: "Grind", 7: "Roast", 8: "Combine", 9: "Serve"
        }

        self.action_space = Discrete(len(self.action_map))

        self.completed_ingredients = set()
        self.curr_ingredient = None
        self.calorie_count = 0
        self.done = False
        self.state = None
        self.reset()
        
 
    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}
        
        action_name = self.action_map[action]
        reward = self.calculate_reward(action_name)
        
        self.state["actions_taken"].append(action_name)

        print(f"\nIngredient: {self.state['ingredient']} | Action: {action_name} | Reward: {reward}")
        print(f"Actions Taken: {self.state['actions_taken']}")

        if action_name == "Combine":
            self.completed_ingredients.add(self.current_ingredient)
            self.state = self.set_next_ingredient()

        return self.state, reward, self.done
    
    def calculate_reward(self, action_name):
        reward = 0
        previous_actions = self.state["actions_taken"]
        correct_prep_method = self.state["prep_method"]

        # 1. Reward "Measure" as the first step
        if len(previous_actions) == 0 and action_name == "Measure":
            reward += 10

        # 2. Penalize actions that break logical constraints
        if action_name == "Wash" and self.state["type"] != "Vegetable":
            reward -= 5  # Only vegetables should be washed

        # 3. Reward correct processing action (Chop/Dice/Shred)
        processing_actions = {"Chop", "Dice", "Shred", "Crush", "Grind"}
        taken_process_steps = set(previous_actions) & processing_actions

        if action_name in processing_actions:
            if action_name == correct_prep_method:
                reward += 10  # Correct processing method
            else:
                reward += 2
            if len(taken_process_steps) > 1:
                reward -= 10  # Penalize multiple processing methods

        # 4. Penalize violating constraints (e.g., allergies)
        if self.current_ingredient in self.constraints.get("allergies", []):
            reward -= 10 

        return reward 

    def set_next_ingredient(self):
        remaining_ingredients = [i for i in self.recipe.keys() if i not in self.completed_ingredients]
        if remaining_ingredients:
            self.current_ingredient = remaining_ingredients[0]
            return {
                "ingredient": self.current_ingredient,
                "type": self.recipe[self.current_ingredient]["type"],
                "prep_method": self.recipe[self.current_ingredient]["prep_method"],
                "actions_taken": []
            }
        else:
            self.done = True
            return {"ingredient": None, "type": None, "prep_method": None, "actions_taken": [], "completed": True}


    def render(self):
        #This is for visualizing the environment
        pass 

    def reset(self):
        self.completed_ingredients = set()
        self.done = False
        self.current_ingredient = None
        self.state = self.set_next_ingredient()
    
    
recipe = {
    "tomato": {"type": "Vegetable", "prep_method": "Dice"},
    "almonds": {"type": "Nuts", "prep_method": "Grind"},
    "sesame_dressing": {"type": "Dressing", "prep_method": "Measure"}
}

constraints = {
    "calories": 300,
    "allergies": ["nuts"]
}

env = SaladEnv(recipe, constraints)
state = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random action (to be replaced with RL policy)
    state, reward, done = env.step(action)
    #print(f"Action: {env.action_map[action]}, Reward: {reward}")



