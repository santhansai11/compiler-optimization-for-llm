"""Reinforcement-learning pass-order search (tabular Q-learning).

State: tuple of passes already applied (canonical order). Actions: apply
one of the remaining passes, or stop. Reward: modeled speedup from the
cost model minus a per-pass complexity cost. The learned policy recommends
which subset of passes to run for a given input graph.
"""

import numpy as np

PASS_KEYS = ["normalize", "canonicalize", "merge", "schedule", "partition"]


def search_pass_order(graph, episodes=80, seed=7, alpha=0.5, gamma=0.9,
                      epsilon_start=0.35, epsilon_end=0.05,
                      num_partitions=2):
    """Search pass subsets with tabular Q-learning.

    Returns ``(best_flags, info)`` where ``best_flags`` maps every pass
    key to a bool for the best-scoring subset found.
    """
    from pipeline import run_pipeline  # local import avoids a cycle
    from utils.cost_model import scheduled_latency_us, sequential_latency_us

    rng = np.random.default_rng(seed)
    q_table = {}
    best = {"reward": -1e9, "flags": None, "sequence": []}
    reward_history = []

    def evaluate(applied):
        flags = {key: key in applied for key in PASS_KEYS}
        optimized, _, _ = run_pipeline(
            graph,
            normalize=flags["normalize"],
            canonicalize=flags["canonicalize"],
            merge=flags["merge"],
            schedule=flags["schedule"],
            partition=flags["partition"],
            num_partitions=num_partitions,
        )
        original_cost = sequential_latency_us(graph)
        optimized_cost = scheduled_latency_us(optimized)
        speedup = original_cost / optimized_cost if optimized_cost else 1.0
        reward = speedup - 0.05 * len(applied)
        return reward, flags

    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
            1.0 - episode / max(1, episodes - 1)
        )
        applied = ()
        total = 0.0

        while True:
            actions = [key for key in PASS_KEYS if key not in applied]
            actions.append("stop")
            q_table.setdefault(applied, {})
            for action in actions:
                q_table[applied].setdefault(action, 0.0)

            if rng.random() < epsilon:
                action = str(rng.choice(actions))
            else:
                action = max(actions, key=lambda a: q_table[applied][a])

            if action == "stop":
                reward, flags = evaluate(applied)
                q_table[applied]["stop"] += alpha * (
                    reward - q_table[applied]["stop"]
                )
                total += reward
                if reward > best["reward"]:
                    best = {"reward": reward, "flags": flags,
                            "sequence": list(applied)}
                break

            next_applied = tuple(
                sorted(applied + (action,),
                       key=lambda key: PASS_KEYS.index(key))
            )
            reward, flags = evaluate(next_applied)
            total += reward
            if reward > best["reward"]:
                best = {"reward": reward, "flags": flags,
                        "sequence": list(next_applied)}

            future_actions = [key for key in PASS_KEYS
                              if key not in next_applied]
            future_actions.append("stop")
            q_table.setdefault(next_applied, {})
            for future_action in future_actions:
                q_table[next_applied].setdefault(future_action, 0.0)
            future = max(q_table[next_applied].values())
            q_table[applied][action] += alpha * (
                reward + gamma * future - q_table[applied][action]
            )
            applied = next_applied

        reward_history.append(float(total))
        if best["flags"] is None:
            reward, flags = evaluate(())
            best = {"reward": reward, "flags": flags, "sequence": []}

    if best["flags"] is None:
        reward, flags = evaluate(())
        best = {"reward": reward, "flags": flags, "sequence": []}

    info = {
        "pass": "rl_pass_order_search",
        "episodes": episodes,
        "seed": seed,
        "best_reward": best["reward"],
        "best_sequence": best["sequence"],
        "best_flags": best["flags"],
        "q_states": len(q_table),
        "reward_history_tail": reward_history[-10:],
    }
    return best["flags"], info
