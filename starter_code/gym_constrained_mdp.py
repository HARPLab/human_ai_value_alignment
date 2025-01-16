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

class SaladMakingEnv(Env):
    def __init__(self, recipe, constraints):
        super(SaladMakingEnv, self).__init__()
        self.recipe = recipe
        self.constraints = constraints
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": 0}
        self.current_stage = 0  # Track progress through recipe stages
        self.step_order = ["Vegetables", "Dressing", "Nuts", "Mix", "Serve"]

    def initialize_ingredients(self):
        # Initialize ingredients based on the recipe
        ingredients = {}
        for name, details in self.recipe.items():
            cls = globals()[details["type"]]
            dietary_restrictions = details.get("dietary_restrictions", [])
            ingredient = cls(name, details["quantity"], dietary_restrictions)

            # Add preparation method to the ingredient's possible actions if specified
            prep_method = details.get("prep_method")
            if prep_method:
                ingredient.possible_actions["Prepare"] = [prep_method]
            else:
                # Set default preparation flow based on ingredient type
                if isinstance(ingredient, Vegetable):
                    ingredient.possible_actions.update({
                        "Wash": ["Wash"],
                        "Combine": ["Combine"]
                    })
                elif isinstance(ingredient, Nuts):
                    ingredient.possible_actions.update({
                        "Roast": ["Roast"],
                        "Combine": ["Combine"]
                    })

            ingredients[name] = ingredient
        return ingredients
    
    def get_stage(self, ingredient):
        # Map ingredient type to stage
        if isinstance(ingredient, Vegetable):
            return "Vegetables"
        elif isinstance(ingredient, Dressing):
            return "Dressing"
        elif isinstance(ingredient, Nuts):
            return "Nuts"
        return None

    def step(self, ingredient_name, action):
        # Get the ingredient
        ingredient = self.ingredients.get(ingredient_name)
        if not ingredient:
            raise ValueError(f"Ingredient {ingredient_name} not found.")

        # Perform action and update state
        try:
            ingredient.update_state(action)
        except ValueError as e:
            return {}, -10, False, {"violations": self.violations}

        # Check constraints
        self.check_constraints(ingredient)

        # Check if stage is complete
        stage_complete = all(ing.is_ready() for ing in self.ingredients.values()
                             if self.get_stage(ing) == self.step_order[self.current_stage])
        if stage_complete:
            self.current_stage += 1

        # Check if all stages are done
        done = self.current_stage >= len(self.step_order)

        return {"state": self.current_stage}, 10, done, {"violations": self.violations}

    def reset(self):
        # Reset state
        self.ingredients = self.initialize_ingredients()
        self.calorie_count = 0
        self.violations = {"calories": 0, "allergies": 0}
        self.current_stage = 0
        return {"state": self.current_stage}

    def check_constraints(self, ingredient):
        # Check calories
        if self.calorie_count + ingredient.quantity > self.constraints["calories"]:
            self.violations["calories"] += 1

        # Check allergies
        for restriction in ingredient.dietary_restrictions:
            if restriction in self.constraints["allergies"]:
                self.violations["allergies"] += 1

    


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