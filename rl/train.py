"""
Phase 1 training script — Q-table on the sailing environment.

Run from the rl/ directory:
    cd rl
    python train.py

A live matplotlib window shows four panels:
  - Episode reward + rolling average
  - Episode length (steps to finish or timeout)
  - Epsilon decay curve
  - Latest trajectory on the course

Every SAVE_EVERY episodes the agent's trajectory is written to
latest_episode.json so watch.html can replay it in the browser.

Every PRINT_EVERY episodes the Q-table is printed to the terminal.
"""

import json
import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from env import SailingEnv, ACTION_NAMES, STATE_NAMES, FINISH_Y, START_Y, CANVAS_W
from q_agent import QAgent

# ── training hyperparameters ──────────────────────────────────────────────────
EPISODES    = 3000
PRINT_EVERY = 250    # print Q-table to terminal
PLOT_EVERY  = 15     # refresh matplotlib window
SAVE_EVERY  = 50     # write latest_episode.json for watch.html
ROLLING_N   = 30     # window for rolling-average reward line

# ── matplotlib setup ──────────────────────────────────────────────────────────

def make_figure():
    plt.ion()
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    axes = {
        'reward':   fig.add_subplot(gs[0, 0]),
        'length':   fig.add_subplot(gs[0, 1]),
        'epsilon':  fig.add_subplot(gs[1, 0]),
        'course':   fig.add_subplot(gs[1, 1]),
    }
    for ax in axes.values():
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b9ab0')
        for spine in ax.spines.values():
            spine.set_color('#2a3a4a')

    axes['reward'].set_title('Episode Reward',  color='#aaccff', fontsize=10)
    axes['length'].set_title('Episode Length',  color='#aaccff', fontsize=10)
    axes['epsilon'].set_title('Epsilon (explore → exploit)', color='#aaccff', fontsize=10)
    axes['course'].set_title('Latest Trajectory', color='#aaccff', fontsize=10)

    fig.suptitle('Sailing RL — Q-table Training', color='#ddeeff', fontsize=13)
    return fig, axes


def update_plots(fig, axes, rewards, lengths, epsilons, trajectory, ep):
    eps_x = list(range(1, len(rewards) + 1))

    # ── reward ────────────────────────────────────────────────────────────────
    ax = axes['reward']
    ax.cla(); ax.set_facecolor('#0d1117')
    ax.plot(eps_x, rewards, color='#334455', linewidth=0.6, label='raw')
    if len(rewards) >= ROLLING_N:
        roll = np.convolve(rewards, np.ones(ROLLING_N)/ROLLING_N, mode='valid')
        ax.plot(range(ROLLING_N, len(rewards)+1), roll,
                color='#44aaff', linewidth=1.8, label=f'avg{ROLLING_N}')
    ax.axhline(0, color='#445566', linewidth=0.5)
    ax.set_xlabel('episode', color='#8b9ab0', fontsize=8)
    ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#aaccff')
    ax.tick_params(colors='#8b9ab0', labelsize=7)
    ax.set_title('Episode Reward', color='#aaccff', fontsize=10)

    # ── episode length ────────────────────────────────────────────────────────
    ax = axes['length']
    ax.cla(); ax.set_facecolor('#0d1117')
    ax.plot(eps_x, lengths, color='#665533', linewidth=0.6)
    if len(lengths) >= ROLLING_N:
        roll = np.convolve(lengths, np.ones(ROLLING_N)/ROLLING_N, mode='valid')
        ax.plot(range(ROLLING_N, len(lengths)+1), roll,
                color='#ffaa44', linewidth=1.8)
    ax.set_xlabel('episode', color='#8b9ab0', fontsize=8)
    ax.set_ylabel('steps', color='#8b9ab0', fontsize=8)
    ax.tick_params(colors='#8b9ab0', labelsize=7)
    ax.set_title('Episode Length (shorter = faster finish)', color='#aaccff', fontsize=10)

    # ── epsilon ───────────────────────────────────────────────────────────────
    ax = axes['epsilon']
    ax.cla(); ax.set_facecolor('#0d1117')
    ax.plot(list(range(1, len(epsilons)+1)), epsilons,
            color='#aa44ff', linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('episode', color='#8b9ab0', fontsize=8)
    ax.tick_params(colors='#8b9ab0', labelsize=7)
    ax.set_title('Epsilon (explore → exploit)', color='#aaccff', fontsize=10)

    # ── course / trajectory ───────────────────────────────────────────────────
    ax = axes['course']
    ax.cla(); ax.set_facecolor('#1a3a5c')
    ax.set_xlim(200, 600); ax.set_ylim(750, 50)   # canvas coords, y inverted
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='#8b9ab0', labelsize=7)
    ax.set_title('Latest Trajectory', color='#aaccff', fontsize=10)

    # Course lines
    ax.axhline(START_Y,  color='#ffffff', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(FINISH_Y, color='#ffcc00', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(205, START_Y  - 8, 'START',  color='#aaaaaa', fontsize=7)
    ax.text(205, FINISH_Y - 8, 'FINISH', color='#ffcc00', fontsize=7)

    if trajectory:
        xs = [p['x'] for p in trajectory]
        ys = [p['y'] for p in trajectory]
        ax.plot(xs, ys, color='#44ff99', linewidth=1.2, alpha=0.8)
        # Mark start and end
        ax.plot(xs[0],  ys[0],  'o', color='#aaaaff', markersize=5)
        ax.plot(xs[-1], ys[-1], 's', color='#ff4444', markersize=5)

    plt.pause(0.001)


# ── episode saving ─────────────────────────────────────────────────────────────

def save_episode(ep, total_reward, steps, finished, trajectory):
    data = {
        'episode':      ep,
        'total_reward': round(float(total_reward), 2),
        'steps':        steps,
        'finished':     finished,
        'trajectory':   trajectory,
    }
    path = os.path.join(os.path.dirname(__file__), 'latest_episode.json')
    with open(path, 'w') as f:
        json.dump(data, f)


# ── main training loop ────────────────────────────────────────────────────────

def main():
    env   = SailingEnv()
    agent = QAgent()

    fig, axes = make_figure()

    rewards_hist  = []
    lengths_hist  = []
    epsilons_hist = []
    latest_traj   = []

    print(f"\nSailing Q-table training — {EPISODES} episodes")
    print(f"State space: {env.observation_space.n} states | "
          f"Action space: {env.action_space.n} actions\n")

    for ep in range(1, EPISODES + 1):
        obs, _      = env.reset()
        total_r     = 0.0
        trajectory  = []
        done = trunc = False

        while not done and not trunc:
            action              = agent.act(obs)
            obs_next, r, done, trunc, info = env.step(action)
            agent.update(obs, action, r, obs_next, done or trunc)

            total_r += r
            obs      = obs_next

            # Save every step's position for the trajectory (sampled to reduce size)
            if ep % SAVE_EVERY == 0 and env.steps % 2 == 0:
                b = info['boat']
                trajectory.append({'x': round(b['x'], 1), 'y': round(b['y'], 1)})

        agent.decay_epsilon()
        finished = done and info['started']

        rewards_hist.append(total_r)
        lengths_hist.append(env.steps)
        epsilons_hist.append(agent.epsilon)

        if ep % SAVE_EVERY == 0:
            latest_traj = trajectory
            save_episode(ep, total_r, env.steps, finished, trajectory)

        # Terminal output
        if ep % 50 == 0:
            roll = np.mean(rewards_hist[-ROLLING_N:])
            fin_rate = sum(
                1 for i in range(max(0, ep-50), ep)
                if i < len(lengths_hist) and lengths_hist[i] < 1500
            )
            print(f"Ep {ep:>4} | R: {total_r:>7.1f} | avg{ROLLING_N}: {roll:>7.1f} | "
                  f"steps: {env.steps:>4} | ε: {agent.epsilon:.3f} | "
                  f"finished: {finished}")

        if ep % PRINT_EVERY == 0:
            agent.print_qtable(top_n=15)
            agent.policy_summary()

        if ep % PLOT_EVERY == 0:
            update_plots(fig, axes, rewards_hist, lengths_hist,
                         epsilons_hist, latest_traj, ep)
            if not plt.fignum_exists(fig.number):
                print("Plot window closed — stopping training.")
                break

    print("\nTraining complete.")
    agent.print_qtable(top_n=20)
    agent.policy_summary()

    plt.ioff()
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
