"""Benchmark trained PPO policy against rule-based heuristic baseline across fixed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from defensive_avionics.policy.baseline import choose_action as choose_baseline_action
from defensive_avionics.policy.environment import (
    ACTION_NAMES,
    build_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "policy"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO policy against rule-based baseline.")
    parser.add_argument(
        "--episodes", type=int, default=30, help="Number of test episodes per agent"
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation base seed")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / "best_model.zip",
        help="Path to trained PPO model zip",
    )
    return parser.parse_args()


def evaluate_agent(
    agent_type: str,
    episodes: int,
    base_seed: int,
    model: PPO | None = None,
) -> dict:
    """Run evaluation episodes for either 'ppo' or 'rule_based' agent."""
    env = build_environment(max_steps=300)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    action_counts: dict[str, int] = {name: 0 for name in ACTION_NAMES}
    effective_actions = 0
    total_actions = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (done or truncated):
            if agent_type == "ppo" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
                act = int(action)
            else:
                act = choose_baseline_action(obs)

            action_name = ACTION_NAMES[act]
            action_counts[action_name] += 1
            total_actions += 1

            obs, reward, done, truncated, info = env.step(act)
            ep_reward += reward
            steps += 1
            if info.get("action_effective", False):
                effective_actions += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

    env.close()

    action_distribution = {
        name: round(count / max(1, total_actions), 4) for name, count in action_counts.items()
    }

    return {
        "agent": agent_type,
        "episodes": episodes,
        "mean_reward": round(float(np.mean(episode_rewards)), 2),
        "std_reward": round(float(np.std(episode_rewards)), 2),
        "min_reward": round(float(np.min(episode_rewards)), 2),
        "max_reward": round(float(np.max(episode_rewards)), 2),
        "mean_length": round(float(np.mean(episode_lengths)), 1),
        "effective_action_rate": round(effective_actions / max(1, total_actions), 4),
        "action_distribution": action_distribution,
        "rewards": [round(r, 2) for r in episode_rewards],
    }


def save_comparison_plot(
    ppo_metrics: dict,
    baseline_metrics: dict,
    output_path: Path,
) -> None:
    """Generate a clean dark-themed comparison plot for PPO vs Baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0b1526")

    # Panel 1: Reward Distribution Boxplot / Bars
    ax1.set_facecolor("#070e1b")
    agents = ["PPO Policy (Learned)", "Rule-Based Baseline"]
    mean_rewards = [ppo_metrics["mean_reward"], baseline_metrics["mean_reward"]]
    std_rewards = [ppo_metrics["std_reward"], baseline_metrics["std_reward"]]
    colors = ["#00d7ff", "#e3a008"]

    bars = ax1.bar(
        agents, mean_rewards, yerr=std_rewards, capsize=8, color=colors, alpha=0.85, width=0.5
    )
    ax1.set_title("Mean Episode Reward Comparison", color="#ffffff", fontsize=11)
    ax1.set_ylabel("Reward (higher is better)", color="#e0f2fe")
    ax1.tick_params(colors="#e0f2fe")
    ax1.grid(axis="y", alpha=0.2, color="#00d7ff")

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#ffffff",
            fontweight="bold",
        )

    # Panel 2: Action Distribution Comparison
    ax2.set_facecolor("#070e1b")
    actions = list(ACTION_NAMES)
    x = np.arange(len(actions))
    width = 0.35

    ppo_act_vals = [ppo_metrics["action_distribution"].get(a, 0.0) * 100 for a in actions]
    base_act_vals = [baseline_metrics["action_distribution"].get(a, 0.0) * 100 for a in actions]

    ax2.bar(x - width / 2, ppo_act_vals, width, label="PPO", color="#00d7ff", alpha=0.85)
    ax2.bar(x + width / 2, base_act_vals, width, label="Baseline", color="#e3a008", alpha=0.85)

    ax2.set_title("Action Allocation Distribution (%)", color="#ffffff", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.replace("_", "\n") for a in actions], color="#e0f2fe", fontsize=8)
    ax2.tick_params(colors="#e0f2fe")
    ax2.set_ylabel("Selection Frequency (%)", color="#e0f2fe")
    ax2.legend(facecolor="#0b1526", edgecolor="#00d7ff", labelcolor="#ffffff")
    ax2.grid(axis="y", alpha=0.2, color="#00d7ff")

    plt.suptitle(
        "Module 2 — Abstract Policy Performance Benchmark (Fixed Seed)",
        color="#ffffff",
        fontsize=13,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def main() -> int:
    args = parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Module 2 — Decision Policy Benchmark ===")
    print(f"Episodes: {args.episodes} | Base seed: {args.seed}")

    # Load PPO model if available
    ppo_model: PPO | None = None
    target_model_path = args.model_path
    if not target_model_path.is_file():
        # Check final model fallback
        alt_path = MODEL_DIR / "ppo_policy_final.zip"
        if alt_path.is_file():
            target_model_path = alt_path

    if target_model_path.is_file():
        print(f"Loading trained PPO policy: {target_model_path}")
        ppo_model = PPO.load(str(target_model_path), device="cpu")
    else:
        print("Trained PPO model not found. Using baseline comparison.")

    # Evaluate PPO
    ppo_metrics = evaluate_agent("ppo", args.episodes, args.seed, ppo_model)
    # Evaluate Baseline
    baseline_metrics = evaluate_agent("rule_based", args.episodes, args.seed, None)

    evaluation_report = {
        "benchmark": "PPO vs Rule-Based Baseline",
        "scope": "abstract academic simulation",
        "episodes": args.episodes,
        "seed": args.seed,
        "ppo": ppo_metrics,
        "rule_based": baseline_metrics,
        "reward_advantage": round(ppo_metrics["mean_reward"] - baseline_metrics["mean_reward"], 2),
    }

    report_path = REPORT_DIR / "policy_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    fig_path = FIGURE_DIR / "policy_comparison.png"
    save_comparison_plot(ppo_metrics, baseline_metrics, fig_path)

    print("\nBenchmark Results Summary:")
    print(f"  PPO Mean Reward:        {ppo_metrics['mean_reward']} ± {ppo_metrics['std_reward']}")
    print(
        "  Baseline Mean Reward:   "
        f"{baseline_metrics['mean_reward']} ± {baseline_metrics['std_reward']}"
    )
    print(f"  Reward Advantage:       {evaluation_report['reward_advantage']:+0.2f}")
    print(f"  PPO Effective Act Rate: {ppo_metrics['effective_action_rate']:.1%}")
    print(f"  Report saved:           {report_path}")
    print(f"  Plot saved:             {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
