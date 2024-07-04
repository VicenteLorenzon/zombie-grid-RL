import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame
import sys

class TLOUGridEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, width, height, num_supplies, num_zombies, num_walls):
        super(TLOUGridEnv, self).__init__()

        self.width = width
        self.height = height
        self.num_supplies = num_supplies
        self.num_zombies = num_zombies
        self.num_walls = num_walls
        self.supplies_taken = 0

        # 0: direita, 1: esquerda, 2: cima, 3: baixo
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(self.width * self.height, )

        self.grid = np.zeros((self.width, self.height), dtype=np.int32)
        self.agent_pos = [0, 0]

        pygame.init()

        self.cell_size = int(900 / self.height)
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption("The Last of Us Grid")

        self.zombies = []
        self.supplies = []
        self.walls = []
        self.door_pos = ()
        self.agent_pos_ini = ()
        while len(self.zombies) < self.num_zombies:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if (x, y) not in self.zombies:
                self.zombies.append((x, y))
        
        while len(self.supplies) < self.num_supplies: 
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if ((x, y) not in self.zombies) and ((x, y) not in self.supplies):
                self.supplies.append((x, y))

        while len(self.walls) < self.num_walls: 
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if ((x, y) not in self.zombies) and ((x, y) not in self.supplies) and ((x, y) not in self.walls):
                self.walls.append((x, y))

        while True:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if ((x, y) not in self.zombies) and ((x, y) not in self.supplies) and ((x, y) not in self.walls):
                self.door_pos = (x, y)
                break

        while True:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if ((x, y) not in self.zombies) and ((x, y) not in self.supplies) and ((x, y) not in self.walls) and ((x, y) != self.door_pos):
                self.agent_pos_ini = (x, y)
                break 

        print(self.supplies)
        print(self.zombies)
        print(self.walls)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.width, self.height), dtype=np.int32)

        # agente
        self.agent_pos = list(self.agent_pos_ini)

        self.supplies_taken = 0

        # suprimentos
        for x, y in self.supplies:
            self.grid[y, x] = 1

        # zumbis
        for x, y in self.zombies:
            self.grid[y, x] = 2 

        # paredes
        for x, y in self.walls:
            self.grid[y, x] = 5                

        # porta
        self.grid[self.door_pos[1], self.door_pos[0]] = 3

        return self._get_obs()

    def _get_obs(self):
        x, y = self.agent_pos
        return [y * self.width + x, self.supplies_taken]
        

    def step(self, action):
        x, y = self.agent_pos
        if (action == 0):  # direita
            x = min(self.width - 1, x + 1)
        elif action == 1:  # esquerda
            x = max(0, x - 1)
        elif action == 2:  # cima
            y = max(0, y - 1)
        elif action == 3:  # baixo
            y = min(self.height - 1, y + 1)

        if self.grid[y, x] == 5:
            x, y = self.agent_pos

        reward = 0

        if self.agent_pos == [x, y]: 
            reward -= 999

        self.agent_pos = [x, y]

        done = False
        if self.grid[y, x] == 1:
            reward += 10
            self.grid[y, x] = 0
            self.supplies_taken += 1
        elif self.grid[y, x] == 2:
            reward += -10
            done = True
        elif self.grid[y, x] == 3:
            done = True
        
        if reward == 0:
            reward = -0.1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))

        colors = {
            0: (255, 255, 255), # nada
            1: (0, 255, 0), # suprimento
            2: (255, 0, 0), # zumbi
            3: (255, 255, 0), # porta
            4: (0, 0, 0), # agente
            5: (150, 150, 150) # parede
        }

        for i in range(self.width):
            for j in range(self.height):
                x = i * self.cell_size
                y = j * self.cell_size
                cell_value = self.grid[j, i]
                pygame.draw.rect(self.screen, colors[cell_value], (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.cell_size, self.cell_size), 1)

        # Desenhar agente
        ax, ay = self.agent_pos
        pygame.draw.rect(self.screen, colors[4], (ax * self.cell_size, ay * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

register(
    id='TLOUGrid',
    entry_point='tlou_grid_env:TLOUGridEnv'
)