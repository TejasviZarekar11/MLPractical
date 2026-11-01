"""
Practical 11: Q-Learning - Grid World

Aim    : Evaluate reinforcement learning with the Q-Learning algorithm on a Grid World, focusing on exploration vs exploitation and the optimal policy.
Dataset: 4 x 4 Grid World (custom environment: start, goal, pit)

Run    : python Practical_11_Q_Learning_GridWorld.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 4x4 Grid World          S = start (0,0)   G = goal (3,3) reward +10
#   S . . .               P = pit  (1,1)    reward -10 (episode ends)
#   . P . .               every other step  reward -1
#   . . . .
#   . . . G
SIZE = 4
START, GOAL, PIT = (0, 0), (3, 3), (1, 1)
ACTIONS = ["U", "D", "L", "R"]
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def step(state, action):
    r, c = state
    dr, dc = MOVES[action]
    r = min(max(r + dr, 0), SIZE - 1)          # stay inside the grid
    c = min(max(c + dc, 0), SIZE - 1)
    new_state = (r, c)
    if new_state == GOAL:
        return new_state, 10, True
    if new_state == PIT:
        return new_state, -10, True
    return new_state, -1, False

# ======================================================================
# Step 2
# ======================================================================
# Q-learning with epsilon-greedy exploration
def train(episodes=1000, alpha=0.1, gamma=0.9,
          eps_start=1.0, eps_min=0.05, eps_decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((SIZE, SIZE, 4))               # Q[row, col, action]
    eps = eps_start
    ep_rewards = []
    for ep in range(episodes):
        s, total, done = START, 0, False
        for _ in range(50):                      # max 50 steps per episode
            if rng.random() < eps:
                a = rng.integers(4)              # explore: random action
            else:
                a = np.argmax(Q[s])              # exploit: best known action
            s2, r, done = step(s, ACTIONS[a])
            # Q-learning update rule
            Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s][a])
            s, total = s2, total + r
            if done:
                break
        eps = max(eps_min, eps * eps_decay)      # reduce exploration over time
        ep_rewards.append(total)
    return Q, ep_rewards

Q, rewards = train()
print("Training finished. Average reward of last 100 episodes:",
      round(np.mean(rewards[-100:]), 2))

# ======================================================================
# Step 3
# ======================================================================
# Learned Q-table (rows = state (row, col), columns = actions)
rows = [(r, c) for r in range(SIZE) for c in range(SIZE)]
q_table = pd.DataFrame([Q[r, c] for r, c in rows], index=[str(s) for s in rows],
                       columns=ACTIONS).round(2)
print(q_table)

# ======================================================================
# Step 4
# ======================================================================
# Optimal policy (best action in every cell)
def show_policy(Q):
    for r in range(SIZE):
        line = []
        for c in range(SIZE):
            if (r, c) == GOAL:
                line.append("G")
            elif (r, c) == PIT:
                line.append("P")
            else:
                line.append(ACTIONS[np.argmax(Q[r, c])])
        print("  ".join(line))

print("Optimal policy (U=up, D=down, L=left, R=right):")
show_policy(Q)

# Follow the greedy policy from the start
s, path, total = START, [START], 0
while s != GOAL and len(path) < 20:
    s, r, done = step(s, ACTIONS[np.argmax(Q[s])])
    path.append(s)
    total += r
    if done:
        break
print("\nPath:", path)
print("Total reward along the path:", total)

# ======================================================================
# Step 5
# ======================================================================
# Exploration vs exploitation: compare strategies
settings = {
    "Low exploration (eps = 0.05 fixed)": dict(eps_start=0.05, eps_min=0.05, eps_decay=1.0),
    "High exploration (eps = 0.9 fixed)": dict(eps_start=0.9, eps_min=0.9, eps_decay=1.0),
    "Decaying eps (1.0 -> 0.05)": dict(eps_start=1.0, eps_min=0.05, eps_decay=0.995),
}
plt.figure(figsize=(9, 4))
for name, kw in settings.items():
    Qs, rw = train(**kw)
    smooth = pd.Series(rw).rolling(30).mean()
    plt.plot(smooth, label=name)
    print(f"{name:38s} avg reward (last 100 episodes) = {np.mean(rw[-100:]):6.2f}")
plt.xlabel("Episode"); plt.ylabel("Reward (moving average)")
plt.title("Effect of exploration strategy")
plt.legend()
plt.grid(True)
plt.show()
