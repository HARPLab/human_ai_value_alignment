# Importing necessary libraries for OpenAI Gym
import gym
from gym import Env, spaces
import numpy as np


class Ingredient:
    def __init__(self, name, available_quantity, dietary_restrictions=None):
        self.name = name
        self.available_quantity = available_quantity
        self.state = "raw"
        self.dietary_restrictions = dietary_restrictions or []
        self.possible_actions = {
            "Measure": ["Measure"],
            "Combine": ["Combine"]
        }
        self.prep_step = 0
        self.requested_quantity = 0
        self.prep_flow = []

    def update_state(self, action):
        # Check if the action is valid for the current step
        current_action_key = list(self.possible_actions.keys())[self.prep_step]
        if action in self.possible_actions[current_action_key]:
            if action == "Measure":
                self.apply_ingredient_effects()
            self.state = action.lower()
            self.prep_step += 1
            return "success"
        else:
            #raise ValueError(f"Invalid action: {action} not allowed at this step.")
            return "invalid"
        
    def apply_ingredient_effects(self):
        if ingredient.requested_quantity <= ingredient.available_quantity:
            ingredient.available_quantity -= ingredient.requested_quantity
            self.calorie_count += ingredient.requested_quantity * ingredient.calories_per_unit
        else:
            self.violations["availability"][ingredient.name] += ingredient.requested_quantity - ingredient.available_quantity

    def is_ready(self):
        # Returns True if all steps are completed
        return self.prep_step >= len(self.prep_flow)

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

AVAILABLE_INGREDIENTS = {
    "tomato": Vegetable("tomato", 2, [], 20),
    "carrot": Vegetable("carrot", 3, [], 25), 
    "lettuce": Vegetable("lettuce", 2, [], 10), 
    "spinach": Vegetable("spinach", 3, [], 15), 
    "almonds": Nuts("almonds", 100, ["nut"], 50), 
    "sesame_dressing": Dressing("sesame_dressing", 100, [], 60),
}

class SaladMakingEnv(Env):
    def __init__(self, recipe, constraints):
        super(SaladMakingEnv, self).__init__()
        self.recipe = recipe
        self.constraints = constraints
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": [], "availability": {}} # how many calories are we over/ under by, which allergies are violated, which ingredients are unavailable
        self.current_step = 0  
        self.step_order = ["Vegetables", "Dressing", "Nuts", "Mix", "Serve"]
    
    def initialize_ingredients(self):
        ingredients = {}
        for ingredient_name, details in self.recipe.items():
            if ingredient_name not in AVAILABLE_INGREDIENTS:
                # Ingredient not available
                self.violations['availability'][ingredient_name] = "Unavailable"
                continue
            
            # Get the ingredient from the available bank
            ingredient = AVAILABLE_INGREDIENTS[ingredient_name]
            ingredient.requested_quantity = details.get("quantity", 0)
            
            # Check availability
            if ingredient.requested_quantity > ingredient.available_quantity:
                excess_request = ingredient.requested_quantity - ingredient.available_quantity
                self.violations['availability'][ingredient_name] = excess_request
                continue

            # Set Preparation flow based on ingredient type
            prep_method = details.get("prep_method")
            if prep_method:
                if "Prepare" in ingredient.possible_actions.keys:
                    if prep_method not in ingredient.possible_actions['Prepare']:
                        raise ValueError(f"Invalid prep method '{prep_method}' for ingredient: '{ingredient_name}'")
                
                if isinstance(ingredient, Vegetable):
                    ingredient.prep_flow = ['Measure', 'Wash', prep_method, "Combine"]

                if isinstance(ingredient, Nuts):
                    ingredient.prep_flow = ['Measure', prep_method, 'Roast', 'Combine']
            else:
                if isinstance(ingredient, Vegetable):
                    ingredient.prep_flow = ["Measure", "Wash", "Combine"]
                elif isinstance(ingredient, Nuts):
                    ingredient.prep_flow = ["Measure", "Roast", "Combine"]
                elif isinstance(ingredient, Dressing):
                    ingredient.prep_flow = ["Measure", "Pour", "Combine"]

            ingredients[ingredient_name] = ingredient

        return ingredients

    def step(self, action):
        reward = 0
        done = False

        current_stage_data = self.get_stage()
        stage = current_stage_data["stage"]
        ingredient_name = current_stage_data.get("ingredient")

        if stage in ["Vegetable", "Dressing", "Nuts"]:
            ingredient = self.ingredients[ingredient_name]
            result = ingredient.update_state(action)

            if result == "success":
                reward += 10
                if ingredient.is_ready():
                    self.current_step += 1
            else:
                reward -= 5
        
        elif stage == "Mix":
            if action == "Mix":
                reward += 10
                self.current_step += 1
            else:
                reward -= 5

        elif stage == "Serve":
            if action == "Serve":
                reward += 10
                done = True
            else:
                reward -= 5

        self.check_constraints()
        if self.violations["calories"] > 0 or self.violations["allergies"] > 0 or self.violations["availability"] > 0:
            reward -= 10

        # Determine if the MDP is complete
        if self.current_step >= len(self.ingredients) + len(self.step_order):
            done = True

        return self._get_observation(), reward, done, {"violations": self.violations}

    def get_stage(self):
        if self.current_step < len(self.ingredients):
            ingredient_names = list(self.ingredients.keys())
            ingredient_name = ingredient_names[self.current_step]
            ingredient = self.ingredients[ingredient_name]

            # Determine stage based on ingredient type
            if isinstance(ingredient, Vegetable):
                stage = "Vegetables"
            elif isinstance(ingredient, Dressing):
                stage = "Dressing"
            elif isinstance(ingredient, Nuts):
                stage = "Nuts"

            return {
                "stage": stage,
                "ingredient": ingredient_name,
                "current_state": ingredient.state,
            }

        elif self.current_step == len(self.ingredients):
            return {"stage": "Mix"}
        elif self.current_step == len(self.ingredients) + 1:
            return {"stage": "Serve"}
        else:
            return {"stage": "Complete"}


    def reset(self):
        # Reset state
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": [], "availability": []}
        self.current_step = 0
        return self.get_observation()
    
    def get_observation(self):
        return {
            "current_stage": self.get_stage(),
            "calorie_count": self.calorie_count,
            "violations": self.violations,
        }

    def check_constraints(self, ingredient):
        if self.calorie_count > self.constraints["calories"]:
            self.violations["calories"] += 1

        for ingredient in self.ingredients.values():
            for restriction in ingredient.dietary_restrictions:
                if restriction in self.constraints["allergies"]:
                    self.violations["allergies"] += restriction
    

recipe = {
    "tomato": {"type": "Vegetable", "quantity": 2, "prep_method": "Dice"},
    "carrot": {"type": "Vegetable", "quantity": 3, "prep_method": "Shred"}, 
    "lettuce": {"type": "Vegetable", "quantity": 1, "prep_method": "Chop"},
    "spinach": {"type": "Vegetable", "quantity": 1},  # just add whole
    "sesame_dressing": {"type": "Dressing", "quantity": 10},
    "almonds": {"type": "Nuts", "quantity": 10, "prep_method": "Grind"},
    "walnuts": {"type": "Nuts", "quantity": 15}  # just roast and add whole
}

constraints = {
    "calories": 300,
    "allergies": ["peanut"]
}

env = SaladMakingEnv(recipe, constraints)
obs = env.reset()

done = False
while not done:
    for ingredient_name, details in recipe.items():
        ingredient = env.ingredients[ingredient_name]
        while not ingredient.is_ready():
            try:
                current_action_key = list(ingredient.possible_actions.keys())[ingredient.current_step]
                action = ingredient.possible_actions[current_action_key][0]
                obs, reward, done, info = env.step(action)
                print(f"Processed {ingredient_name}: {action}, Reward: {reward}")
            except IndexError:
                break
    print(f"Violations: {info['violations']}")