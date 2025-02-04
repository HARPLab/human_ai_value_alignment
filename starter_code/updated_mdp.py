from gym import Env
from gym.spaces import Discrete, Dict
import numpy as np
import random


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
            5: "Crush", 6: "Grind", 7: "Roast", 8: "Combine"
        }

        self.action_space = Discrete(len(self.action_map))

        self.state_space = len(recipe.keys()) * len(self.action_map)
        self.observation_space = Discrete(self.state_space)

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

        return self.encode_state(), reward, self.done, {}
    
    def calculate_reward(self, action_name):
        reward = 0
        previous_actions = self.state["actions_taken"]
        correct_prep_method = self.state["prep_method"]

        # 1. Reward "Measure" as the first step
        if len(previous_actions) == 0:
            if action_name == "Measure":
                reward += 30
            else:
                reward -= 30

        # 2. Ensure "Combine" is the last step 
        if action_name == "Combine":
            if correct_prep_method == None and "Measure" in previous_actions:
                    if self.state["type"] == "Nuts":
                        if "Roast" in previous_actions: 
                            reward += 20  
                        else:
                            reward -= 10  # didn't roast nut before adding
                    else:
                        reward += 20
            else:
                reward -= 15 # too many missed steps

        # 3. Enforce "Wash" only after "Measure" for vegetables
        if action_name == "Wash":
            if self.state["type"] == "Vegetable":
                if "Measure" in previous_actions:
                    reward += 20  # Correct placement
                else:
                    reward -= 10  # Incorrect order
            else:
                reward -= 30  # Not a vegetable

        # 4. Penalize unnecessary actions
        if action_name in previous_actions:
            reward -= 100  # Avoid repeating actions unnecessarily

        # 5. Reward correct processing action (Chop/Dice/Shred)
        processing_actions = {"Chop", "Dice", "Shred", "Crush", "Grind"}
        taken_process_steps = set(previous_actions) & processing_actions

        if action_name in processing_actions:
            if action_name == correct_prep_method:
                reward += 10  # Correct processing method
            else:
                reward += 2
            if len(taken_process_steps) > 1:
                reward -= 100  # Penalize multiple processing methods

        # 6. Penalize violating constraints (e.g., allergies)
        if self.current_ingredient in self.constraints.get("allergies", []):
            reward -= 30 
        
        # Dressing should only be measured and combined
        if self.state["type"] == "Dressing" and action_name not in ["Measure", "Combine"]:
            reward -= 100 

        return reward 
    
    def encode_state(self):
        ingredient_index = list(self.recipe.keys()).index(self.current_ingredient) if self.current_ingredient else len(self.recipe)
        state_index = ingredient_index * len(self.action_map) + len(self.state["actions_taken"])
        return state_index % num_states  # Prevent out-of-bounds errors


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
        return self.encode_state()

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 500

# Initialize Q-table
num_states = 1000  # Arbitrary large number for state space
num_actions = 9  # Number of actions
Q = np.zeros((num_states, num_actions))

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

for episode in range(episodes):
    state = env.reset()  # Ensure this returns a valid state
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit best known action

        next_state, reward, done, _ = env.step(action)

        # Q-learning update
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

    if episode % 50 == 0:
        print(f"Episode {episode} completed, current Q-table values: {Q[:5]}")  # Print sample Q-values for debugging


print("Training finished.")
print("Sample Q-values after training:")
print(Q[:5])  # Print the first few rows to check if updates are happening

    
'''state = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random action (to be replaced with RL policy)
    state, reward, done = env.step(action)
    #print(f"Action: {env.action_map[action]}, Reward: {reward}")'''


state = env.reset()
done = False

while not done:
    action = np.argmax(Q[state, :])  # Use learned policy
    state, reward, done, _ = env.step(action)
    print(f"Chosen Action: {env.action_map[action]}, Reward: {reward}")




