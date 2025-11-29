import numpy as np
import random

# Types de terrain
EMPTY = 0
OBSTACLE = 1  # Cratère ou rocher (Pénalité)
SAMPLE = 2    # Régolithe (Récompense)
BASE = 3      # Base pour recharger/déposer

class LunarEnv:
    def __init__(self, rows=10, cols=10, n_obstacles=15, n_samples=5):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        self.start_pos = (0, 0)
        self.rover_pos = self.start_pos
        self.energy = 100
        self.max_energy = 100
        self.payload = 0
        self.sensor_range = 2 # Le rover voit à 2 cases de distance
        
        self._generate_world(n_obstacles, n_samples)

    def _generate_world(self, n_obstacles, n_samples):
        # Placer la base
        self.grid[0, 0] = BASE
        
        # Placer les obstacles aléatoirement
        for _ in range(n_obstacles):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos:
                self.grid[r, c] = OBSTACLE
                
        # Placer les échantillons (samples)
        for _ in range(n_samples):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos and self.grid[r, c] != OBSTACLE:
                self.grid[r, c] = SAMPLE

    def get_observation(self):
        """
        Retourne une vue locale autour du rover (Partial Observability).
        Les zones hors de portée ou hors de la grille sont masquées (-1).
        """
        r, c = self.rover_pos
        obs_grid = np.full((self.rows, self.cols), -1) # -1 = Inconnu
        
        r_min = max(0, r - self.sensor_range)
        r_max = min(self.rows, r + self.sensor_range + 1)
        c_min = max(0, c - self.sensor_range)
        c_max = min(self.cols, c + self.sensor_range + 1)
        
        obs_grid[r_min:r_max, c_min:c_max] = self.grid[r_min:r_max, c_min:c_max]
        
        return {
            "local_grid": obs_grid,
            "energy": self.energy,
            "payload": self.payload,
            "position": self.rover_pos
        }

    def step(self, action):
        """
        Actions: 0: Haut, 1: Bas, 2: Gauche, 3: Droite, 4: Collecter
        Retourne: next_state, reward, done
        """
        r, c = self.rover_pos
        reward = -1 # Pénalité de temps (step cost) pour encourager l'efficacité
        done = False
        
        # Consommation d'énergie pour le mouvement
        energy_cost = 1
        
        # Dynamique de mouvement (avec stochasticité possible ici)
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.rows - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.cols - 1, c + 1)
        elif action == 4: # Collecter
            energy_cost = 5 # Coûte plus cher de forer
            if self.grid[r, c] == SAMPLE:
                reward += 50 # Grande récompense
                self.grid[r, c] = EMPTY
                self.payload += 1
            else:
                reward -= 5 # Pénalité si on fore pour rien

        # Vérification du terrain
        if self.grid[r, c] == OBSTACLE:
            reward -= 20 # Grosse pénalité
            energy_cost += 10 # Coute cher de traverser un obstacle
            # Optionnel: Le rover reste bloqué ou retourne en arrière ?
        
        self.rover_pos = (r, c)
        self.energy -= energy_cost
        
        if self.energy <= 0:
            done = True
            reward -= 100 # Mort du rover
            
        return self.get_observation(), reward, done

    def reset(self):
        self.rover_pos = self.start_pos
        self.energy = self.max_energy
        self.payload = 0
        return self.get_observation()
