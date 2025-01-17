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
        self.current_step = 0  # Tracks progress through preparation steps
        self.requested_quantity = 0

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

AVAILABLE_INGREDIENTS = {
    "tomato": Vegetable("tomato", 100, [], 20),
    "carrot": Vegetable("carrot", 100, [], 25), 
    "lettuce": Vegetable("lettuce", 100, [], 10), 
    "spinach": Vegetable("spinach", 100, [], 15), 
    "almonds": Nuts("almonds", 10, ["nut"], 50), 
    "sesame_dressing": Dressing("sesame_dressing", 100, [], 60),
}

class SaladMakingEnv(Env):
    def __init__(self, recipe, constraints):
        super(SaladMakingEnv, self).__init__()
        self.recipe = recipe
        self.constraints = constraints
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": [], "availability": []} # how many calories are we over/ under by, which allergies are violated, which ingredients are unavailable
        self.current_step = 0  
        self.step_order = ["Vegetables", "Dressing", "Nuts", "Mix", "Serve"]

    def initialize_ingredients(self):
        # Initialize ingredients based on the recipe
        ingredients = {}
        for ingredient_name, details in self.recipe.items():
            ingredient.requested_quantity = details.get("quantity", 0)
            if ((ingredient_name not in AVAILABLE_INGREDIENTS) or
                (ingredient.requested_quantity > ingredient.available_quantity)):
                self.violations['availability'] += [ingredient_name]

            else:
                ingredient = AVAILABLE_INGREDIENTS[ingredient_name]
                prep_method = details.get("prep_method")

            if prep_method:
                if prep_method not in ingredient.possible_actions:
                    raise ValueError(f"Invalid prep method '{prep_method}' for ingredient: '{ingredient_name}'")
                
                if isinstance(ingredient, Vegetable):
                    ingredient.set_prep_flow(['Measure', 'Wash', prep_method, "Combine"])

                if isinstance(ingredient, Nuts):
                    ingredient.set_prep_flow('Measure', prep_method, 'Roast', 'Combine')
            else:
                if isinstance(ingredient, Vegetable):
                    ingredient.set_preparation_flow(["Measure", "Wash", "Combine"])
                elif isinstance(ingredient, Nuts):
                    ingredient.set_preparation_flow(["Measure", "Roast", "Combine"])
                elif isinstance(ingredient, Dressing):
                    ingredient.set_preparation_flow(["Measure", "Pour", "Combine"])


            ingredients.reset()
            ingredients[ingredient_name] = ingredient
        return ingredients


    def step(self, action):
        reward = 0
        done = False

        current_stage_data = self.get_stage()
        stage = current_stage_data["stage"]

        if stage in ["Vegetable", "Dressing", "Nuts"]:
            current_ingredient = current_stage_data["ingredient"]
            result = self.ingredients[current_ingredient].perform_action(action)

            if result == "success":
                reward += 10
                if self.ingredients[current_ingredient].is_preparation_complete():
                    # Move to the next ingredient or stage
                    self.current_step += 1
            elif result == "invalid":
                reward -= 5
            elif result == "constraint_violation":
                reward -= 20

        elif stage == "mix":
            if action == "Mix":
                reward += 20
                self.current_step += 1
            else:
                reward -= 5

        elif stage == "serve":
            if action == "Serve":
                reward += 30
                done = True
            else:
                reward -= 5

        # Check for constraint violations
        self._check_constraints()
        if self.violations["calories"] > 0 or self.violations["allergies"] > 0 or self.violations["availability"] > 0:
            reward -= 10

        # Determine if the MDP is complete
        if self.current_step >= len(self.ingredients) + len(self.step_order):
            done = True

        return self._get_observation(), reward, done, {"violations": self.violations}

    def get_stage(self):
        if self.current_step < len(self.ingredients):
            ingredient_names = list(self.ingredients.keys())
            ingredient_index = self.current_step

            # Categorize the current ingredient into its respective stage
            ingredient_name = ingredient_names[ingredient_index]
            if isinstance(self.ingredients[ingredient_name], Vegetable):
                return {
                    "stage": "vegetables",
                    "ingredient": ingredient_name,
                    "required_flow": self.ingredients[ingredient_name].required_flow,
                    "current_action_index": self.ingredients[ingredient_name].current_action_index,
                    "current_state": self.ingredients[ingredient_name].state,
                }
            elif isinstance(self.ingredients[ingredient_name], Dressing):
                return {
                    "stage": "dressing",
                    "ingredient": ingredient_name,
                    "required_flow": self.ingredients[ingredient_name].required_flow,
                    "current_action_index": self.ingredients[ingredient_name].current_action_index,
                    "current_state": self.ingredients[ingredient_name].state,
                }
            elif isinstance(self.ingredients[ingredient_name], Nuts):
                return {
                    "stage": "nuts",
                    "ingredient": ingredient_name,
                    "required_flow": self.ingredients[ingredient_name].required_flow,
                    "current_action_index": self.ingredients[ingredient_name].current_action_index,
                    "current_state": self.ingredients[ingredient_name].state,
                }

        elif self.current_step == len(self.ingredients):
            return {"stage": "mix"}
        elif self.current_step == len(self.ingredients) + 1:
            return {"stage": "serve"}
        else:
            return {"stage": "complete"}


    def reset(self):
        # Reset state
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": [], "availability": []}
        self.current_step = 0
        return self.get_observation()
    
    def apply_ingredient_effects(self, ingredient):
        if ingredient.requested_quantity <= ingredient.available_quantity:
            ingredient.available_quantity -= ingredient.requested_quantity
            self.calorie_count += ingredient.requested_quantity * ingredient.calories_per_unit
        else:
            self.violations["availability"] += 1
    
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
    "tomato": {"type": "Vegetable", "quantity": 20, "prep_method": "Dice"},
    "carrot": {"type": "Vegetable", "quantity": 30, "prep_method": "Shred"}, 
    "lettuce": {"type": "Vegetable", "quantity": 50, "prep_method": "Chop"},
    "spinach": {"type": "Vegetable", "quantity": 50},  # just add whole
    "sesame_dressing": {"type": "Dressing", "quantity": 30},
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
                obs, reward, done, info = env.step(ingredient_name, action)
                print(f"Processed {ingredient_name}: {action}, Reward: {reward}")
            except IndexError:
                break
    print(f"Violations: {info['violations']}")