import numpy as np
import random

# Terrain types
EMPTY = 0
CRATER = 1
SAMPLE = 2
BASE = 3
SLOPE = 4

class LunarEnv:
    
    def __init__(self, rows=10, cols=10, n_obstacles=15, n_samples=5,
                 max_payload=3, slippage_chance=0.2, sensor_range=2):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        self.start_pos = (0, 0)
        self.base_pos = self.start_pos
        self.rover_pos = self.start_pos
        self.max_energy = 100
        self.max_payload = max_payload
        self.sensor_range = sensor_range
        self.slippage_chance = slippage_chance
        
        self.actions = {
            0: 'Up',
            1: 'Down',
            2: 'Left',
            3: 'Right',
            4: 'Collect',
            5: 'Recharge'
        }
        self.action_space_size = len(self.actions)

        # Dictionary mapping sample coordinates → sample reward in [40, 60]
        self.sample_rewards = {}

        self._generate_world(n_obstacles, n_samples)
        self.reset()

    def _generate_world(self, n_obstacles, n_samples):
        """
        Generates the world grid with:
        - Base at (0,0)
        - Obstacles, randomly CRATER or SLOPE
        - A random number of samples (1 to n_samples), each with reward in [40, 60]
        """
        self.grid = np.zeros((self.rows, self.cols))
        self.sample_rewards = {}

        # Place base
        self.grid[self.start_pos] = BASE
        
        # Place obstacles
        for _ in range(n_obstacles):
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                obstacle_type = CRATER if random.random() < 0.5 else SLOPE
                self.grid[r, c] = obstacle_type
                
        # Choose a random number of samples
        if n_samples > 0:
            actual_samples = random.randint(1, n_samples)
        else:
            actual_samples = 0
        
        # Place samples and assign rewards
        for _ in range(actual_samples):
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                self.grid[r, c] = SAMPLE
                self.sample_rewards[(r, c)] = random.randint(40, 60)

    def get_feature_vector(self):
        """
        Returns the partially observable state representation:
        - Local terrain patch (sensor window)
        - Normalized energy
        - Normalized payload
        """
        r, c = self.rover_pos
        R = self.sensor_range
        
        # 1. Create local sensor window
        view_size = 2 * R + 1
        local_view = np.full((view_size, view_size), -1.0)  # Unknown = -1
        
        # Determine boundaries of visible region in the actual grid
        r_start_grid = max(0, r - R)
        r_end_grid = min(self.rows, r + R + 1)
        c_start_grid = max(0, c - R)
        c_end_grid = min(self.cols, c + R + 1)
        
        # Map grid region into the local view array (with padding)
        r_start_local = r_start_grid - (r - R)
        r_end_local = view_size - ((r + R + 1) - r_end_grid)
        c_start_local = c_start_grid - (c - R)
        c_end_local = view_size - ((c + R + 1) - c_end_grid)

        # Copy visible grid into sensor window
        local_view[r_start_local:r_end_local, c_start_local:c_end_local] = \
            self.grid[r_start_grid:r_end_grid, c_start_grid:c_end_grid]
            
        # 2. Append normalized energy and payload
        norm_energy = self.energy / self.max_energy
        norm_payload = self.payload / self.max_payload
        
        return np.concatenate([
            local_view.flatten(),
            [norm_energy],
            [norm_payload]
        ])

    def _determine_actual_move(self, desired_action):
        """
        Applies slippage: with probability slippage_chance,
        the rover moves in a random direction instead of the chosen one.
        No slippage for Collect/Recharge.
        """
        if desired_action not in [0, 1, 2, 3]:
            return desired_action
        
        if random.random() < self.slippage_chance:
            return random.choice([0, 1, 2, 3])
        else:
            return desired_action

    def step(self, action_idx):
        """
        Executes one action step.
        Returns: (feature_vector, reward, done)
        """
        r, c = self.rover_pos
        reward = -1  # Time penalty
        done = False
        energy_cost = 1  # Base cost per step
        
        # Determine actual action after slippage
        actual_action = self._determine_actual_move(action_idx)

        new_r, new_c = r, c
        
        # Movement actions
        if actual_action == 0: new_r = max(0, r - 1)
        elif actual_action == 1: new_r = min(self.rows - 1, r + 1)
        elif actual_action == 2: new_c = max(0, c - 1)
        elif actual_action == 3: new_c = min(self.cols - 1, c + 1)
        
        # Collect action (only triggered if agent intended to Collect)
        elif action_idx == 4:
            energy_cost = 5
            if self.grid[r, c] == SAMPLE and self.payload < self.max_payload:
                sample_reward = self.sample_rewards.pop((r, c), 50)
                reward += sample_reward
                self.grid[r, c] = EMPTY
                self.payload += 1
            else:
                reward -= 5
        
        # Recharge action
        elif action_idx == 5:
            energy_cost = 5  # Time spent waiting
            if self.grid[r, c] == BASE:
                energy_gain = min(100, self.max_energy - self.energy)
                energy_cost = -energy_gain
            else:
                reward -= 10
        
        # Update position after move
        self.rover_pos = (new_r, new_c)
        r, c = self.rover_pos

        # Penalties for entering obstacles
        if action_idx in [0, 1, 2, 3]:
            if self.grid[r, c] == CRATER:
                reward -= 25
                energy_cost += 15
            elif self.grid[r, c] == SLOPE:
                reward -= 15
                energy_cost += 5
        
        # Update energy
        self.energy = min(self.max_energy, self.energy - energy_cost)
        
        # Energy depleted → episode ends
        if self.energy <= 0:
            done = True
            if self.rover_pos == self.base_pos:
                # Success: rover returned home before shutdown
                reward += 50
            else:
                # Failure: rover died away from home
                reward -= 100
        
        return self.get_feature_vector(), reward, done

    def reset(self):
        """ Resets rover position, energy, and payload for a new episode. """
        self.rover_pos = self.start_pos
        self.energy = self.max_energy
        self.payload = 0
        return self.get_feature_vector()

