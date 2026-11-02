# 11 Evaluate the performance of reinforcement learning using the Q-Learning algorithm for a simple game or decision-making problem (e.g., Tic-Tac-Toe or Grid World),focusing on exploration–exploitation strategies and the optimal policy.
                                                                               
#Cell 1: Import Libraries

import numpy as np
import random
import matplotlib.pyplot as plt

#Cell 2: Create Grid World

grid_size = 4

start = (0, 0)
goal = (3, 3)

actions = [0, 1, 2, 3]

print("Grid Size:", grid_size, "x", grid_size)
print("Start:", start)
print("Goal:", goal)


#Cell 3: Define Environment

def move(state, action):

    row, col = state

    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col -= 1
    elif action == 3:
        col += 1

    row = max(0, min(row, 3))
    col = max(0, min(col, 3))

    new_state = (row, col)

    if new_state == goal:
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return new_state, reward, done



#Cell 4: Initialize Q-Table

q_table = np.zeros((16, 4))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0

episodes = 5000

print(q_table)

#Cell 5: Train the Q-Learning Agent

rewards = []

for episode in range(episodes):

    state = start
    total_reward = 0

    for step in range(50):

        state_number = state[0] * 4 + state[1]

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state_number])

        new_state, reward, done = move(state, action)

        new_number = new_state[0] * 4 + new_state[1]

        old_value = q_table[state_number, action]

        next_value = np.max(q_table[new_number])

        q_table[state_number, action] = old_value + learning_rate * (
            reward + discount_factor * next_value - old_value
        )

        state = new_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)

    epsilon = max(0.01, epsilon * 0.995)

print("Training Completed!")
print("Final Epsilon:", epsilon)

#Cell 6: Display Learning Performance

plt.plot(rewards)

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Performance")

plt.show()


#Cell 7: Display Learned Optimal Policy

symbols = ['↑', '↓', '←', '→']

for row in range(4):

    for col in range(4):

        state_number = row * 4 + col

        if (row, col) == goal:
            print("G", end="  ")
        else:
            best_action = np.argmax(q_table[state_number])
            print(symbols[best_action], end="  ")

    print()


#Cell 8: Evaluate the Learned Policy

state = start
total_steps = 0

print("Agent Path:")

while state != goal and total_steps < 20:

    print(state, end=" -> ")

    state_number = state[0] * 4 + state[1]

    action = np.argmax(q_table[state_number])

    state, reward, done = move(state, action)

    total_steps += 1

print(goal)

if state == goal:
    print("Goal Reached!")
else:
    print("Goal Not Reached!")

print("Total Steps:", total_steps)
