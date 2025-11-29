import numpy as np
import random

# Types de terrain
EMPTY = 0
OBSTACLE = 1  
SAMPLE = 2    
BASE = 3      

class LunarEnv:
    
    def __init__(self, rows=10, cols=10, n_obstacles=15, n_samples=5, max_payload=3, slippage_chance=0.2, sensor_range=2):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        self.start_pos = (0, 0)
        self.rover_pos = self.start_pos
        self.max_energy = 100
        self.max_payload = max_payload
        self.sensor_range = sensor_range # Rayon du capteur
        self.slippage_chance = slippage_chance
        
        self.actions = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Collect', 5: 'Recharge'}
        self.action_space_size = len(self.actions)

        self._generate_world(n_obstacles, n_samples)
        self.reset()

    def _generate_world(self, n_obstacles, n_samples):
        # ... (Logique de génération de la grille inchangée) ...
        self.grid = np.zeros((self.rows, self.cols))
        self.grid[0, 0] = BASE
        
        for _ in range(n_obstacles):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                self.grid[r, c] = OBSTACLE
                
        for _ in range(n_samples):
            r, c = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
            if (r, c) != self.start_pos and self.grid[r, c] == EMPTY:
                self.grid[r, c] = SAMPLE


    def get_feature_vector(self):
        """
        Génère un vecteur de caractéristiques (phi(s)) pour l'agent LFA.
        Ceci est la représentation de l'état partiallement observable.
        """
        r, c = self.rover_pos
        R = self.sensor_range
        
        # 1. Caractéristiques de l'environnement local (La vue du capteur)
        
        # Définir la fenêtre du capteur (-R à +R) -> (2R+1) x (2R+1)
        view_size = 2 * R + 1
        local_view = np.full((view_size, view_size), -1.0) # -1.0 pour Inconnu
        
        # Coordonnées des bords
        r_start_grid = max(0, r - R)
        r_end_grid = min(self.rows, r + R + 1)
        c_start_grid = max(0, c - R)
        c_end_grid = min(self.cols, c + R + 1)
        
        # Coordonnées dans la matrice local_view (padding)
        r_start_local = r_start_grid - (r - R)
        r_end_local = view_size - ((r + R + 1) - r_end_grid)
        c_start_local = c_start_grid - (c - R)
        c_end_local = view_size - ((c + R + 1) - c_end_grid)

        # Copier la zone visible
        local_view[r_start_local:r_end_local, c_start_local:c_end_local] = \
            self.grid[r_start_grid:r_end_grid, c_start_grid:c_end_grid]
            
        # 2. Concaténation des caractéristiques
        
        # Normalisation des caractéristiques continues (pour l'énergie/payload)
        norm_energy = self.energy / self.max_energy
        norm_payload = self.payload / self.max_payload
        
        # Vectorisation: [local_grid_flattened, norm_energy, norm_payload]
        feature_vector = np.concatenate([
            local_view.flatten(),
            [norm_energy],
            [norm_payload]
        ])
        
        return feature_vector

    def _determine_actual_move(self, desired_action):
        """
        Détermine l'action de mouvement réelle en fonction de la stochasticité.
        """
        # L'action est-elle un mouvement (0, 1, 2, 3) ?
        if desired_action not in [0, 1, 2, 3]:
            return desired_action # Si c'est Collecter/Recharger, pas de glissement
        
        if random.random() < self.slippage_chance:
            # Glissement : Choisir une action aléatoire parmi les 4 mouvements
            actual_move = random.choice([0, 1, 2, 3])
            return actual_move
        else:
            return desired_action # Mouvement souhaité effectué

    def step(self, action_idx):
        """
        Actions: 0: Haut, 1: Bas, 2: Gauche, 3: Droite, 4: Collecter, 5: Recharger
        Retourne: feature_vector, reward, done
        """
        r, c = self.rover_pos
        reward = -1 # Pénalité de temps (step cost)
        done = False
        
        energy_cost = 1
        
        # --- 1. DÉTERMINER L'ACTION RÉELLE (STOCHASTICITÉ) ---
        actual_action = self._determine_actual_move(action_idx)

        new_r, new_c = r, c
        
        # --- 2. EXÉCUTION DE L'ACTION RÉELLE ---
        if actual_action == 0: new_r = max(0, r - 1)
        elif actual_action == 1: new_r = min(self.rows - 1, r + 1)
        elif actual_action == 2: new_c = max(0, c - 1)
        elif actual_action == 3: new_c = min(self.cols - 1, c + 1)
        
        # Action COLLECTER (seulement si l'action désirée était 4)
        elif action_idx == 4: # On pénalise l'énergie si l'agent voulait forer
            energy_cost = 5
            if self.grid[r, c] == SAMPLE and self.payload < self.max_payload:
                reward += 50
                self.grid[r, c] = EMPTY
                self.payload += 1
            else:
                reward -= 5
        
        # Action RECHARGER (seulement si l'action désirée était 5)
        elif action_idx == 5:
            energy_cost = 5 # Coût de l'attente
            if self.grid[r, c] == BASE:
                energy_gain = min(100, self.max_energy - self.energy) 
                energy_cost = -energy_gain # Gain net
            else:
                reward -= 10
        
        # --- 3. MISE À JOUR DE L'ÉTAT ET VÉRIFICATIONS ---
        
        # Mise à jour de la position APRÈS le mouvement
        self.rover_pos = (new_r, new_c)
        r, c = self.rover_pos

        # Vérification du terrain (si l'action était un mouvement)
        if action_idx in [0, 1, 2, 3]: 
            if self.grid[r, c] == OBSTACLE:
                reward -= 20 
                energy_cost += 10 
        
        # Gestion de l'énergie (min pour le plafond max_energy)
        self.energy = min(self.max_energy, self.energy - energy_cost)
        
        if self.energy <= 0:
            done = True
            reward -= 100 
            
        # Remplacement de get_state_dict() par get_feature_vector()
        return self.get_feature_vector(), reward, done

    def reset(self):
        self.rover_pos = self.start_pos
        self.energy = self.max_energy
        self.payload = 0
        # Retourne le vecteur de features initial
        return self.get_feature_vector()
