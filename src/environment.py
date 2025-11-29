import numpy as np
import random

# Types de terrain
EMPTY = 0
OBSTACLE = 1  
SAMPLE = 2    
BASE = 3      

class LunarEnv:
    # --- AJUSTEMENT DES PARAMÈTRES D'INITIALISATION ---
    def __init__(self, rows=10, cols=10, n_obstacles=15, n_samples=5, max_payload=3):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        self.start_pos = (0, 0)
        self.rover_pos = self.start_pos
        self.max_energy = 100
        self.max_payload = max_payload # Ajout de max_payload
        self.sensor_range = 2
        
        self.actions = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Collect', 5: 'Recharge'} # Ajout de l'action Recharge
        self.action_space_size = len(self.actions)

        self._generate_world(n_obstacles, n_samples)
        self.reset() # Appel à reset pour initialiser l'état actuel

    def _generate_world(self, n_obstacles, n_samples):
        # ... (Logique inchangée pour la génération de la grille) ...
        self.grid = np.zeros((self.rows, self.cols)) # Réinitialisation pour la génération
        self.grid[0, 0] = BASE
        
        # Placer les obstacles aléatoirement
        for _ in range(n_obstacles):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                self.grid[r, c] = OBSTACLE
                
        # Placer les échantillons (samples)
        for _ in range(n_samples):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                self.grid[r, c] = SAMPLE


    def get_state_dict(self):
        """
        Retourne l'état complet du rover pour l'agent (état non-observable complet si le Q-Learning est Tabulaire).
        """
        return {
            "energy": self.energy,
            "payload": self.payload,
            "position": self.rover_pos,
            "grid_type": self.grid[self.rover_pos] # Type de terrain actuel
        }

    def step(self, action_idx):
        """
        Actions: 0: Haut, 1: Bas, 2: Gauche, 3: Droite, 4: Collecter, 5: Recharger
        Retourne: next_state_dict, reward, done
        """
        r, c = self.rover_pos
        reward = -1 # Pénalité de temps
        done = False
        
        new_r, new_c = r, c
        energy_cost = 1

        # Mouvement Cartésien
        if action_idx == 0: new_r = max(0, r - 1)
        elif action_idx == 1: new_r = min(self.rows - 1, r + 1)
        elif action_idx == 2: new_c = max(0, c - 1)
        elif action_idx == 3: new_c = min(self.cols - 1, c + 1)
        
        # Action COLLECTER
        elif action_idx == 4:
            energy_cost = 5 # Coût du forage
            if self.grid[r, c] == SAMPLE and self.payload < self.max_payload:
                reward += 50
                self.grid[r, c] = EMPTY # Le régolithe est enlevé
                self.payload += 1
            else:
                reward -= 5 # Pénalité: fore pour rien ou payload plein
        
        # Action RECHARGER
        elif action_idx == 5:
            energy_cost = -10 # Gain d'énergie (Négatif pour représenter le gain)
            if self.grid[r, c] == BASE:
                # Gain max: (max_energy - current_energy), limité à 100
                energy_gain = min(100, self.max_energy - self.energy) 
                energy_cost = -energy_gain # Gain net
            else:
                reward -= 10 # Pénalité: essaye de recharger hors de la base
                energy_cost = 5 # Coût normal de l'attente

        # Mise à jour de la position APRÈS le mouvement
        self.rover_pos = (new_r, new_c)
        r, c = self.rover_pos # nouvelle position

        # Vérification du terrain (si l'action était un mouvement)
        if action_idx in [0, 1, 2, 3]:
            if self.grid[r, c] == OBSTACLE:
                reward -= 20 
                energy_cost += 10 # Coût supplémentaire pour l'obstacle
        
        # Gestion de l'énergie (Note: Si action == 5, energy_cost est négatif, donc on ajoute)
        self.energy = min(self.max_energy, self.energy - energy_cost)
        
        if self.energy <= 0:
            done = True
            reward -= 100 
            
        return self.get_state_dict(), reward, done

    def reset(self):
        self.rover_pos = self.start_pos
        self.energy = self.max_energy
        self.payload = 0
        return self.get_state_dict()
