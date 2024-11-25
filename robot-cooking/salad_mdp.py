import srl_example_setup
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.agents.AgentClass import Agent

import copy 


'''
Things to think about:
- recipe ingredients vs. available ingredients 
- recipe steps
- include dietary restrictions/ preferences 
- how to keep track or errors/ deviations from the recipes (maybe have a substitution count?)
'''

class SaladMakingMDP(MDP):

    actions = ["check_ingredients", "measure", "clarify", "substitute", 
                 "cut", "clean", "mix", "add_dressing", "serve"]

    init_state = SaladState(
            ingredients={"lettuce": "unwashed", "avocado": "unwashed", "cucumber": "unwashed"}, # dressing should technically be part of ingredients too
            stage="initial",
            tools=["knife", "bowl", "serving spoon"]
    )

    def transition_func(state, action):
            return self._transition_logic(state, action)

        def reward_func(state, action, next_state):
            return self._reward_logic(state, action, next_state)

        super().__init__(actions, transition_func, reward_func, init_state)

    def _transition_logic(self, state, action):
        """
        Handles state transitions based on the current state and action.
        """
        next_state = copy.deepcopy(state)  
        ingredients = next_state.data["ingredients"]
        stage = next_state.data["stage"]

        if action == "check_ingredients":
            if "missing" in ingredients.values():
                next_state.data["stage"] = "clarify"
            else:
                next_state.data["stage"] = "measuring"
        
        elif action == "measure":
            for ingredient, status in ingredients.items():
                if status == "raw":
                    ingredients[ingredient] = "measured"
            next_state.data["stage"] = "preparation"
        
        elif action == "cut":
            for ingredient, status in ingredients.items():
                if status == "measured":
                    ingredients[ingredient] = "cut"
            next_state.data["stage"] = "mixing"
        
        elif action == "mix":
            next_state.data["stage"] = "dressing"
        
        elif action == "add_dressing":
            next_state.data["stage"] = "serving"
        
        elif action == "serve":
            next_state.data["stage"] = "done"
            next_state.set_terminal(True)

        return next_state

     def _reward_logic(self, state, action, next_state):
       pass 

class SaladState(State):
    def __init__(self, ingredients, stage, tools, is_terminal=False):
        """
        Args:
            ingredients (dict): Dictionary where keys are ingredient names and values are their statuses.
            stage (str): Current stage in the process (e.g., 'initial', 'measuring', 'mixing').
            tools (list): List of available tools.
            is_terminal (bool): Whether this is a terminal state.
        """
        super().__init__(data={"ingredients": ingredients, "stage": stage, "tools": tools}, is_terminal=is_terminal)

    def features(self):
        """
        Summary:
            Converts the state into a numerical feature vector for use in RL.
        """
        return np.array(list(self.data["ingredients"].values()) + [self.data["stage"]])

    def __str__(self):
        return f"Stage: {self.data['stage']}, Ingredients: {self.data['ingredients']}, Tools: {self.data['tools']}"

class SaladAgent(Agent):
    def __init__(self, name, actions):
        super().__init__(name, actions)

    def act(self, state, reward):
        """
        Decides the next action based on the current state.
        """
        stage = state.data["stage"]

        if stage == "initial":
            return "check_ingredients"
        elif stage == "measuring":
            return "measure"
        elif stage == "preparation":
            return "cut"
        elif stage == "mixing":
            return "mix"
        elif stage == "dressing":
            return "add_dressing"
        elif stage == "serving":
            return "serve"
        return "clarify" 