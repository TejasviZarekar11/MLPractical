# 11 Evaluate the performance of reinforcement learning using the Q-Learning algorithm for a simple game or decision-making problem (e.g., Tic-Tac-Toe or Grid World),focusing on exploration–exploitation strategies and the optimal policy.
import numpy as np
import matplotlib.pyplot as plt

# Create Grid World
rows = 4
cols = 4

actions = ['Up', 'Down', 'Left', 'Right']

# Create Q-table
Q = np.zeros((16, 4))

# Set parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

start = 0
goal = 15

# Function to move agent
def move(state, action):
    row = state // cols
    col = state % cols

    if action == 0 and row > 0:
        row -= 1
    elif action == 1 and row < rows - 1:
        row += 1
    elif action == 2 and col > 0:
        col -= 1
    elif action == 3 and col < cols - 1:
        col += 1

    return row * cols + col

# Train agent
rewards = []

for episode in range(episodes):
    state = start
    total_reward = 0

    for step in range(50):

        # Exploration and exploitation
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        next_state = move(state, action)

        # Reward
        if next_state == goal:
            reward = 10
        else:
            reward = -1

        # Q-value update
        if next_state == goal:
            future_reward = 0
        else:
            future_reward = np.max(Q[next_state])

        Q[state, action] += alpha * (
            reward + gamma * future_reward - Q[state, action]
        )

        total_reward += reward
        state = next_state

        if state == goal:
            break

    rewards.append(total_reward)

# Display Q-table
print("Learned Q-Table:")
print(Q)

# Display optimal policy
print("\nOptimal Policy:")

for i in range(rows):
    for j in range(cols):
        state = i * cols + j

        if state == goal:
            print("G", end="   ")
        else:
            best_action = np.argmax(Q[state])
            print(actions[best_action], end="  ")

    print()

# Plot learning progress
plt.plot(rewards)
plt.title("Q-Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
