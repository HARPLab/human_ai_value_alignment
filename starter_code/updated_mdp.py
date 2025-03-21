from gym import Env
from gym.spaces import Discrete, Dict
import numpy as np
import random


# constraints we want to manage: skill limitation, time limitation, calorie, availability, allergies 

AVAILABLE_INGREDIENTS = {
    "tomato": {"type": "Vegetable", "calories": 20, "quantity": 3, "allergy": {"tomato"}}, 
    "carrot": {"type": "Vegetable", "calories": 15, "quantity": 2, "allergy": {"carrot"}}, 
    "lettuce": {"type": "Vegetable", "calories": 10, "quantity": 1, "allergy": {"lettuce"}}, 
    "spinach": {"type": "Vegetable", "calories": 12, "quantity": 2, "allergy": {"spinach"}}, 
    "almonds": {"type": "Nuts", "calories": 20, "quantity": 100, "allergy": {"almond", "nuts"}}, 
    "sesame_dressing": {"type": "Dressing", "calories": 30, "quantity": 100, "allergy": {"sesame"}},
    "peanuts": {"type": "Nuts", "calories": 20, "quantity": 100, "allergy": {"peanut", "nuts"}}
}

AVAILABLE_INGREDIENTS = {
    "tomato": {"type": "Vegetable", "calories": 20, "quantity": 3, "allergy": {"tomato"}}, 
    "carrot": {"type": "Vegetable", "calories": 15, "quantity": 2, "allergy": {"carrot"}}, 
    "lettuce": {"type": "Vegetable", "calories": 10, "quantity": 1, "allergy": {"lettuce"}}, 
    "spinach": {"type": "Vegetable", "calories": 12, "quantity": 2, "allergy": {"spinach"}}, 
    "almonds": {"type": "Nuts", "calories": 50, "quantity": 100, "allergy": {"almond", "nuts"}}, 
    "sesame_dressing": {"type": "Dressing", "calories": 30, "quantity": 100, "allergy": {"sesame"}},
    "peanuts": {"type": "Nuts", "calories": 60, "quantity": 100, "allergy": {"peanut", "nuts"}}, 
    "cucumber": {"type": "Vegetable", "calories": 10, "quantity": 2,  "allergy": {"cucumber"}}, 
    "onion": {"type": "Vegetable", "calories": 15, "quantity": 2, "allergy": {"onion"}},
    "cheese": {"type": "Dairy", "calories": 90, "quantity": 50, "allergy": {"cheese", "dairy"}},
    "croutons": {"type": "Grain", "calories": 80, "quantity": 20, "allergy": {"croutons", "gluten"}}
}

class SaladEnv(Env): 
    def __init__(self, recipe, constraints):
        super(SaladEnv, self).__init__()

        # Recipe and Constraints are user defined
        self.recipe = recipe 
        self.constraints = constraints 

        self.violations = {"calories": 0, "allergies": [], "availability":{}}

        # Actions we can take: Measure, Wash, Chop, Dice, Shred, Crush, Grind, Roast, Combine
        self.action_map = {
            0: "Measure", 1: "Wash", 2: "Chop", 3: "Dice", 4: "Shred",
            5: "Crush", 6: "Grind", 7: "Roast", 8: "Combine"
        }

        self.action_space = Discrete(len(self.action_map))
        self.state_space = len(recipe.keys()) * len(self.action_map)
        self.observation_space = Discrete(self.state_space)
        self.completed_ingredients = set()
        self.warnings = []

        self.current_calories = 0
        self.available_ingredients = AVAILABLE_INGREDIENTS
        self.reset()
        
 
    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}
        
        action_name = self.action_map[action]
        reward = self.calculate_reward(action_name)
        
        self.state["actions_taken"].append(action_name)

        if action_name == "Combine":
            self.completed_ingredients.add(self.current_ingredient)
            print(f"\nIngredient: {self.state['ingredient']} | Action: {action_name} | Reward: {reward}")
            print(f"Actions Taken: {self.state['actions_taken']}")
            self.state = self.set_next_ingredient()

        return self.encode_state(), reward, self.done, {"warnings": self.warnings, "violations": self.violations}

    def calculate_reward(self, action_name):
        reward = 0
        previous_actions = self.state["actions_taken"]
        ingredient_type = self.state["type"]
        correct_prep_method = self.state["prep_method"]
        ingredient_name = self.state["ingredient"]

        # Prevent using unavailable ingredients
        if not self.available_ingredients.get(ingredient_name, True):
            return -500

        # Ensure first action is "Measure"
        if len(previous_actions) == 0:
            return 100 if action_name == "Measure" else -100

        # Prevent actions before measuring
        if "Measure" not in previous_actions:
            return -50
        
        # Ensure only vegetables are washed
        if action_name == "Wash":
            if ingredient_type == "Vegetable":
                reward += 50 if "Wash" not in previous_actions else -25
            else:
                reward -= 100  # Harsh penalty for washing non-vegetables

        if action_name == "Roast":
            if ingredient_type == "Nuts":
                if any(act in previous_actions for act in {"Grind", "Crush", "Dice"}):
                    reward += 75  # Normal reward for roasting
                elif "Roast" not in previous_actions:
                    reward += 125  # Encourage roasting before processing
                else:
                    reward -= 10  # Small penalty for repeating roast
            else:
                reward -= 100  # Harsh penalty for roasting non-nuts

        # Ensure dressing only has "Measure" and "Combine"
        if ingredient_type == "Dressing":
            if action_name not in {"Measure", "Combine"}:
                reward -= 100  # Penalize invalid actions for dressing

        # Reward correct processing action
        if action_name in {"Chop", "Dice", "Shred", "Crush", "Grind"}:
            if action_name == correct_prep_method:
                reward += 75  
            else:
                reward -= 50  

        # Penalize repeating an action but not too harshly
        if action_name in previous_actions:
            reward -= 50  

        # Ensure "Wash" happens for vegetables
        if ingredient_type == "Vegetable" and action_name == "Wash":
            reward += 50 if "Wash" not in previous_actions else -25

        # Ensure "Roast" happens for nuts
        if ingredient_type == "Nuts" and action_name == "Roast":
            reward += 75 if "Roast" not in previous_actions else -25

        # Calorie constraint tracking
        ingredient_calories = self.available_ingredients[ingredient_name].get("calories", 0)
        self.current_calories += ingredient_calories  # Track total calories

        calorie_limit = self.constraints.get("calories", float("inf"))
        if self.current_calories > calorie_limit:
            overshoot = self.current_calories - calorie_limit
            self.violations["calories"] = overshoot  # Log the violation
            #reward -= 50 + (overshoot * 5)  # Scale penalty based on overshoot
            self.warnings.append(f"Warning: Calorie limit exceeded by {overshoot} calories!")

        # Allergy Constraint Handling
        if ingredient_name in self.available_ingredients:
            allergens = self.available_ingredients[ingredient_name].get("allergy", set())
            user_allergies = set(self.constraints.get("allergies", []))
            
            if allergens & user_allergies:  # Intersection means violation
                if ingredient_name not in self.violations["allergies"]:
                    self.violations["allergies"].append(ingredient_name)  # Log violation
                #reward -= 100  # Apply penalty
                self.warnings.append(f"Warning: Allergy violation! {ingredient_name} contains allergens: {allergens & user_allergies}")


        # Reward completing ingredient correctly
        if action_name == "Combine" and set(previous_actions) >= self.get_required_actions():
            reward += 100  

        return reward 
    
    def get_required_actions(self):
        required = {"Measure"}
        if self.state["type"] == "Vegetable":
            required.add("Wash")
        elif self.state["type"] == "Nuts":
            required.add("Roast")
        return required
    
    def encode_state(self):
        ingredient_index = list(self.recipe.keys()).index(self.current_ingredient)
        
        # Convert action history to binary vector (0 if not taken, 1 if taken)
        action_vector = [1 if action in self.state["actions_taken"] else 0 for action in self.action_map.values()]
        
        return ingredient_index * len(self.action_map) + int("".join(map(str, action_vector)), 2)

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
        self.current_calories = 0
        self.warnings = []  # Reset warnings to avoid accumulation
        self.state = self.set_next_ingredient()
        self.available_ingredients = AVAILABLE_INGREDIENTS
        return self.encode_state()
    
# Training parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
EPISODES = 2000

def train_agent(env, episodes=EPISODES):
    num_states = 2 ** len(env.action_map) * len(env.recipe.keys())
    num_actions = len(env.action_map)
    Q = np.zeros((num_states, num_actions))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            
            next_state, reward, done, info = env.step(action)
            
            # Q-learning update
            Q[state, action] = Q[state, action] + LEARNING_RATE * (
                reward + DISCOUNT * np.max(Q[next_state, :]) - Q[state, action]
            )
            
            state = next_state
            
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    return Q



# Example usage
recipe = {
    "tomato": {"type": "Vegetable", "prep_method": "Dice", "quantity": 2},
    "almonds": {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
    "sesame_dressing": {"type": "Dressing", "prep_method": None, "quantity": 1},
    "peanuts" : {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
}

constraints = {
    "calories": 300,
    "allergies": ["peanut"]
}

env = SaladEnv(recipe, constraints)
Q_table = train_agent(env)
print("\nTraining Done!\n")

# Test the trained policy
def test_policy(env, Q_table):
    state = env.reset()
    done = False
    total_reward = 0
    episode_warnings= []
    
    while not done:
        action = np.argmax(Q_table[state, :])
        state, reward, done, info = env.step(action)
        total_reward += reward

        if "warnings" in info:
            episode_warnings.extend(info["warnings"])
        
    if episode_warnings:
        print("\n⚠️ Warnings:")
        for warning in set(episode_warnings):
            print(f" - {warning}")
    else:
        print("\n No warnings issued!")

    print("\n Violations:")
    for key, value in env.violations.items():
        if isinstance(value, list) and value:
            print(f" - {key.capitalize()}: {', '.join(value)}")
        elif isinstance(value, dict) and value:
            print(f" - {key.capitalize()}: {value}")
        elif value:
            print(f" - {key.capitalize()}: {value}")

    print(f"\nTotal reward: {total_reward}")


test_policy(env, Q_table)




