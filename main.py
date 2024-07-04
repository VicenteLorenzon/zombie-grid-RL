import time
from turtle import heading, width
import numpy as np
import random
import gymnasium as gym
import tlou_grid_env
from tlou_grid_env import TLOUGridEnv

height = int(input("Altura: "))
width = int(input("Largura: "))
supplies = int(input("Suprimentos: "))
zombies = int(input("Zumbis: "))
walls = int(input("Paredes: "))

env = gym.make('TLOUGrid', width=width, height=height, num_supplies=supplies, num_zombies=zombies, num_walls=walls)

# parametros
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.997
num_episodes = 50000
max_steps = 200

q_table = np.zeros((height*width, supplies+1, env.action_space.n))

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state[0], state[1]])

for episode in range(num_episodes):
    state = env.reset()
    state = tuple(state)
    total_reward = 0
    
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)
        
        q_value = q_table[state][action]
        best_next_q = np.max(q_table[next_state])
        new_q_value = q_value + alpha * (reward + gamma * best_next_q - q_value)
        q_table[state][action] = new_q_value
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if episode % 100 == 0:
        print(f"Episodio: {episode}, Recompensa: {total_reward}")

if True:
    state = env.reset()
    state = tuple(state)
    total_reward = 0
    
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)
        
        state = next_state
        total_reward += reward

        env.render()        
        time.sleep(0.5)

        if done:
            break

env.close()