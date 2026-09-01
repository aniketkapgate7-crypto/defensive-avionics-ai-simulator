"""Train PPO in the normalized academic environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from defensive_avionics.policy.environment import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    build_environment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "policy"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO in the abstract environment.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5_000,
    )
    return parser.parse_args()


def create_monitored_environment(
    filename: str,
    max_steps: int,
    seed: int,
) -> Monitor:
    environment = Monitor(
        build_environment(max_steps=max_steps),
        filename=str(LOG_DIR / filename),
    )
    environment.reset(seed=seed)
    return environment


def main() -> int:
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    training_environment = create_monitored_environment(
        "training",
        args.max_steps,
        args.seed,
    )
    evaluation_environment = create_monitored_environment(
        "evaluation",
        args.max_steps,
        args.seed + 1,
    )

    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=args.eval_frequency,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_policy",
    )

    callbacks = CallbackList(
        [
            evaluation_callback,
            checkpoint_callback,
        ]
    )

    model = PPO(
        policy="MlpPolicy",
        env=training_environment,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(LOG_DIR / "tensorboard"),
        seed=args.seed,
        device=args.device,
    )

    print("Scope: abstract academic simulation")
    print("Observations:", OBSERVATION_NAMES)
    print("Actions:", ACTION_NAMES)
    print("Requested timesteps:", args.total_timesteps)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    final_model_path = MODEL_DIR / "ppo_policy_final"
    model.save(final_model_path)

    metadata = {
        "algorithm": "PPO",
        "scope": "abstract academic simulation",
        "total_timesteps": args.total_timesteps,
        "max_episode_steps": args.max_steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "observation_names": OBSERVATION_NAMES,
        "action_names": ACTION_NAMES,
        "final_model": "ppo_policy_final.zip",
        "best_model": "best_model.zip",
    }

    with (MODEL_DIR / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    training_environment.close()
    evaluation_environment.close()

    print("\nPPO training completed")
    print(
        "Final model:",
        final_model_path.with_suffix(".zip"),
    )
    print(
        "Best model:",
        MODEL_DIR / "best_model.zip",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
