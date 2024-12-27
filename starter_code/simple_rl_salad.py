from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.planning.ValueIterationClass import ValueIteration

# Define states as a list of string labels
states = [
    "Start",
    "Measuring",
    "Cleaning",
    "Cutting",
    "Mixing",
    "Dressing",
    "Serving",
    "Task Complete"
]

# Define actions as general robot actions
actions = ["Measure", "Clean", "Cut", "Mix", "Add Dressing", "Serve"]

# Define rewards: positive rewards for correct actions, negative for violations
def reward_func(state, action):
    if state.name == "Task Complete":
        return 100  # Task completion reward
    if action in ["Measure", "Clean", "Cut", "Mix", "Add Dressing", "Serve"]:
        return 10  # Positive reward for recipe adherence
    return -10  # Penalty for invalid actions or constraint violations

# Define transitions: A dictionary to simulate the FSM
def transition_func(state, action):
    transitions = {
        "Start": {"Measure": "Measuring", "Clarify": "Clarification"},
        "Measuring": {"Clean": "Cleaning", "Clarify": "Clarification"},
        "Cleaning": {"Cut": "Cutting", "Clarify": "Clarification"},
        "Cutting": {"Mix": "Mixing", "Clarify": "Clarification"},
        "Mixing": {"Add Dressing": "Dressing", "Clarify": "Clarification"},
        "Dressing": {"Serve": "Serving", "Clarify": "Clarification"},
        "Serving": {"Complete": "Task Complete", "Clarify": "Clarification"},
        "Clarification": {"Substitute": "Substitute", "Measure": "Measuring"},
        "Substitute": {"Measure": "Measuring"},
    }
    # Return the next state based on the current state and action
    next_state = transitions.get(state.name, {}).get(action, state.name)
    return State(next_state)

# Create the MDP
initial_state = State("Start")
salad_mdp = MDP(actions, transition_func, reward_func, initial_state)

# Run Value Iteration to find the optimal policy
vi = ValueIteration(salad_mdp)
vi.run_vi()
optimal_policy = vi.get_policy()

# Display the optimal policy
for state in optimal_policy:
    print(f"State: {state}, Optimal Action: {optimal_policy[state]}")
