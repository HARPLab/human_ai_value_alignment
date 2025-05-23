import matplotlib.pyplot as plt
import numpy as np
from updated_mdp import SaladEnv, train_agent, test_policy
import pickle

with open("trained_q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Multiple test recipes
TEST_RECIPES = [
    {
        "recipe": {
            "tomato": {"type": "Vegetable", "prep_method": "Dice", "quantity": 2},
            "almonds": {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
            "sesame_dressing": {"type": "Dressing", "prep_method": None, "quantity": 1},
            "peanuts": {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
        },
        "constraints": {
            "calories": 300,
            "allergies": ["peanut"]
        }
    },
    {
        "recipe": {
            "carrot": {"type": "Vegetable", "prep_method": "Shred", "quantity": 2},
            "cucumber": {"type": "Vegetable", "prep_method": "Dice", "quantity": 1},
            "onion": {"type": "Vegetable", "prep_method": "Chop", "quantity": 1},
            "croutons": {"type": "Grain", "prep_method": None, "quantity": 5}
        },
        "constraints": {
            "calories": 250,
            "allergies": []
        }
    },
    {
        "recipe": {
            "spinach": {"type": "Vegetable", "prep_method": "Wash", "quantity": 2},
            "cheese": {"type": "Dairy", "prep_method": None, "quantity": 10},
            "tomato": {"type": "Vegetable", "prep_method": "Dice", "quantity": 1}
        },
        "constraints": {
            "calories": 200,
            "allergies": ["cheese"]
        }
    }
]

# Number of trials per recipe
NUM_TRIALS = 10
results = {}

for i, test_case in enumerate(TEST_RECIPES):
    recipe = test_case["recipe"]
    constraints = test_case["constraints"]

    print(f"\nTraining on Recipe {i+1} with constraints {constraints}")
    env = SaladEnv(recipe, constraints)
    print("\nTraining Done!\n")

    rewards = []
    violation_counts = {}

    for _ in range(NUM_TRIALS):
        state = env.reset()
        done = False
        total_reward = 0
        episode_warnings = []

        while not done:
            action = np.argmax(Q_table[state, :])
            state, reward, done, info = env.step(action)
            total_reward += reward

            if "warnings" in info:
                episode_warnings.extend(info["warnings"])

        # Track results
        rewards.append(total_reward)

        # Track warnings and violations
        for key, value in env.violations.items():
            if isinstance(value, list) and value:
                violation_counts[key] = violation_counts.get(key, 0) + len(value)
            elif isinstance(value, dict) and value:
                violation_counts[key] = violation_counts.get(key, 0) + len(value)
            elif value:
                violation_counts[key] = violation_counts.get(key, 0) + 1

    results[f"Recipe {i+1}"] = {"rewards": rewards, "violations": violation_counts}

    print(f"\nResults for Recipe {i+1}:")
    print(f"Average Reward: {np.mean(rewards)}")
    print(f"Violation Counts: {violation_counts}")

# Plot results
plt.figure(figsize=(10, 6))
for recipe, data in results.items():
    plt.plot(range(NUM_TRIALS), data["rewards"], label=recipe)

plt.xlabel("Trial Number")
plt.ylabel("Total Reward")
plt.title("Performance of Salad Making Recipes")
plt.legend()
plt.show()
