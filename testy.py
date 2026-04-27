import math
import os
import argparse
from collections import deque

import numpy as np
import pygame
import torch

import train as MRS

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
XML_PATH    = MRS.XML_PATH
MODEL_PATH  = MRS.STAGE2_MODEL   # "models/stage2_both.pth"
MAX_STEPS   = 100_000
DETERMINISTIC = True             # use mean action (no sampling noise)


# ─────────────────────────────────────────────────────────────────────────────
# Demo loop
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(xml_path: str, model_path: str, viz: MRS.Visualizer):
    # Load policy
    policy = MRS.ActorCritic()
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()
    print(f"Loaded weights from '{model_path}'")

    env         = MRS.CartDoublePendulumEnv(xml_path, stage=2)
    obs         = env.reset()
    ep_reward   = 0.0
    episode     = 1
    total_steps = 0
    recent      = deque(maxlen=20)
    last_action = 0.0

    print(f"Running demo — press the window's ✕ to quit.\n")

    while total_steps < MAX_STEPS:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            dist, value = policy(obs_t)
            # Deterministic: take the mean; stochastic: dist.sample()
            action = dist.mean if DETERMINISTIC else dist.sample()
            logp   = dist.log_prob(action).sum(-1)

        action_np   = action.numpy()[0]
        last_action = float(action_np[0])

        obs, reward, done, _ = env.step(action_np)
        ep_reward   += reward
        total_steps += 1

        ep_just_done    = False
        ep_reward_done  = 0.0

        if done:
            recent.append(ep_reward)
            ep_just_done   = True
            ep_reward_done = ep_reward
            mean20         = float(np.mean(recent))

            print(f"  Episode {episode:4d} | Steps {total_steps:8,} "
                  f"| EpRew {ep_reward:8.1f} | Mean20 {mean20:8.1f}")

            ep_reward = 0.0
            episode  += 1
            obs       = env.reset()

        mean20 = float(np.mean(recent)) if recent else 0.0

        viz.update(env.state, {
            "stage":          2,
            "total_steps":    total_steps,
            "episode":        episode,
            "ep_reward":      ep_reward,
            "ep_reward_done": ep_reward_done,
            "mean20":         mean20,
            "action":         last_action,
            "ep_just_done":   ep_just_done,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Demo: Cart Double Inverted Pendulum")
    ap.add_argument("--xml",   default=XML_PATH,   help="Path to MuJoCo XML")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to .pth weights file")
    ap.add_argument("--stage1", action="store_true",
                    help="Load Stage-1 weights instead of Stage-2")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample from the policy distribution instead of using the mean")
    args = ap.parse_args()

    if args.stage1:
        args.model = MRS.STAGE1_MODEL

    global DETERMINISTIC
    if args.stochastic:
        DETERMINISTIC = False

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found: '{args.model}'\n"
            f"Train first with:  python train.py"
        )

    viz = MRS.Visualizer()

    try:
        run_demo(args.xml, args.model, viz)
    except SystemExit:
        pass          # user closed the Pygame window — clean exit
    finally:
        viz.close()

    print("\nDemo finished.")


if __name__ == "__main__":
    main()