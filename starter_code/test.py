import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
from updated_mdp import SaladEnv  # assumes train_agent and test_policy not needed here


TEST_RECIPES = [
    {
        "recipe": {
            "tomato": {"type": "Vegetable", "prep_method": "Dice", "quantity": 2},
            "almonds": {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
            "sesame_dressing": {"type": "Dressing", "prep_method": None, "quantity": 1},
            "peanuts": {"type": "Nuts", "prep_method": "Grind", "quantity": 10},
        },
        "constraints": {"calories": 300, "allergies": ["peanut"]}
    },
    {
        "recipe": {
            "carrot": {"type": "Vegetable", "prep_method": "Shred", "quantity": 2},
            "cucumber": {"type": "Vegetable", "prep_method": "Dice", "quantity": 1},
            "onion": {"type": "Vegetable", "prep_method": "Chop", "quantity": 1},
            "croutons": {"type": "Grain", "prep_method": None, "quantity": 5}
        },
        "constraints": {"calories": 250, "allergies": []}
    },
    {
        "recipe": {
            "spinach": {"type": "Vegetable", "prep_method": "Wash", "quantity": 2},
            "cheese": {"type": "Dairy", "prep_method": None, "quantity": 10},
            "tomato": {"type": "Vegetable", "prep_method": "Dice", "quantity": 1}
        },
        "constraints": {"calories": 200, "allergies": ["cheese"]}
    }
]


def run_trial(env, Q_table, seed=None):
    """
    Executes one trial in the environment using the greedy policy from the Q-table.

    Args:
        env (SaladEnv): The initialized environment.
        Q_table (np.ndarray): The pre-trained Q-table.
        seed (int): Optional random seed for reproducibility.

    Returns:
        total_reward (float): Cumulative reward obtained in the episode.
        env.violations (dict): Constraint violations collected during the trial.
    """

    if seed is not None:
        np.random.seed(seed)
        #env.seed(seed)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q_table[state, :])
        state, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward, env.violations


def count_violations(violations_dict, aggregate_counts):
    """
    Tallies violations from a single trial into the running total.

    Args:
        violations_dict (dict): Violation info returned from the environment.
        aggregate_counts (defaultdict): Running total of each violation type.
    """
    for k, v in violations_dict.items():
        if isinstance(v, (list, dict)):
            count = len(v)
        else:
            count = int(bool(v))
        aggregate_counts[k] += count


def evaluate_policy(Q_table, num_trials=10):
    """
    Evaluates the Q-table policy on each test recipe multiple times.

    Args:
        Q_table (np.ndarray): Trained Q-table.
        num_trials (int): Number of trials to run per recipe.

    Returns:
        results (dict): Mapping from recipe names to rewards and violations.
    """
    results = {}
    for i, test_case in enumerate(TEST_RECIPES):
        print(f"\nEvaluating Recipe {i+1} with constraints: {test_case['constraints']}")
        env = SaladEnv(test_case["recipe"], test_case["constraints"])

        rewards = []
        violation_counts = defaultdict(int)

        for t in range(num_trials):
            reward, violations = run_trial(env, Q_table, seed=t)
            rewards.append(reward)
            count_violations(violations, violation_counts)

        results[f"Recipe {i+1}"] = {
            "rewards": rewards,
            "violations": dict(violation_counts)
        }

        print(f"Avg Reward: {np.mean(rewards):.2f}")
        print(f"Violations: {dict(violation_counts)}")

    return results


def plot_rewards(results):
    """
    Plots the average total reward (with standard deviation) for each recipe.

    Args:
        results (dict): Output from evaluate_policy().
    """
    means = [np.mean(data["rewards"]) for data in results.values()]
    stds = [np.std(data["rewards"]) for data in results.values()]
    labels = list(results.keys())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=5, color='skyblue')
    plt.ylabel("Average Total Reward")
    plt.title("Agent Performance on Salad Recipes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_violations(results):
    """
    Stacks violation counts for each recipe by type.

    Args:
        results (dict): Output from evaluate_policy().
    """
    violation_types = set()
    for data in results.values():
        violation_types.update(data["violations"].keys())

    violation_matrix = {v: [] for v in violation_types}
    for data in results.values():
        for v in violation_types:
            violation_matrix[v].append(data["violations"].get(v, 0))

    x = np.arange(len(results))
    bottom = np.zeros(len(results))
    plt.figure(figsize=(10, 6))

    for v_type, vals in violation_matrix.items():
        plt.bar(x, vals, bottom=bottom, label=v_type)
        bottom += np.array(vals)

    plt.xticks(x, list(results.keys()))
    plt.ylabel("Violation Count")
    plt.title("Constraint Violations Across Recipes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(qtable_path: str, num_trials: int):
    """
    Loads the Q-table, evaluates it, and plots the results.

    Args:
        qtable_path (str): Path to the saved Q-table file.
        num_trials (int): Number of evaluation trials per recipe.
    """
    with open(qtable_path, "rb") as f:
        Q_table = pickle.load(f)

    results = evaluate_policy(Q_table, num_trials=num_trials)
    plot_rewards(results)
    plot_violations(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default="trained_q_table.pkl",
                        help="Path to the trained Q-table pickle file")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of evaluation trials per recipe")
    args = parser.parse_args()

    main(args.qtable, args.trials)
