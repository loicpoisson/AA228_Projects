import numpy as np
import random

class QLearningAgent:
    """
    Agent Q-Learning pour le Rover Lunaire.
    Utilise Q-Learning Tabulaire pour le moment.
    """
    def __init__(self, action_space_size, grid_rows, grid_cols, max_energy, max_payload, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.action_space_size = action_space_size
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'escompte
        self.epsilon = epsilon  # Taux d'exploration initiale
        self.epsilon_min = 0.01 # Taux d'exploration minimum
        self.epsilon_decay = 0.9999 # Taux de décroissance de l'exploration
        
        # --- Définition de l'Espace d'État (pour indexer la table Q) ---
        # L'énergie est trop grande (0-100), il faut la discrétiser.
        # Simplification: discrétisation de l'énergie en 5 niveaux
        self.energy_levels = 5 
        
        # Simplification: discrétisation du payload (nombre d'échantillons)
        self.payload_levels = max_payload + 1 # De 0 à max_payload
        
        # Taille totale de la table Q
        self.q_table_shape = (grid_rows, grid_cols, self.energy_levels, self.payload_levels, action_space_size)
        self.q_table = np.zeros(self.q_table_shape)
        
        # Mémorisation des paramètres pour la conversion d'état
        self._max_energy = max_energy
        self._energy_step = max_energy / self.energy_levels

    def _discretize_state(self, state):
        """
        Convertit l'état continu/complexe en index discrets pour la table Q.
        Input: Dictionnaire d'état de l'environnement
        Output: Tuple (row_idx, col_idx, energy_idx, payload_idx)
        """
        r, c = state['position']
        energy = state['energy']
        payload = state['payload']
        
        # 1. Discrétisation de l'énergie: 
        # Exemple: [100, 80) -> 4, [80, 60) -> 3, ..., [0, 20) -> 0
        energy_idx = int(energy // self._energy_step)
        # S'assurer que l'indice maximum est correct même si energy = max_energy
        if energy == self._max_energy:
            energy_idx = self.energy_levels - 1
        
        # 2. Payload (déjà discret)
        payload_idx = payload
        
        return (r, c, energy_idx, payload_idx)

    def choose_action(self, state):
        """
        Stratégie Epsilon-Greedy pour choisir une action.
        """
        state_idx = self._discretize_state(state)
        
        # 1. Exploration (random)
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        
        # 2. Exploitation (best action from Q-table)
        q_values = self.q_table[state_idx]
        # Si plusieurs actions ont la même valeur max, choisir aléatoirement parmi elles
        best_action = np.argmax(q_values)
        return int(best_action)

    def learn(self, state, action, reward, next_state, done):
        """
        Mise à jour de la table Q.
        """
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)

        # 1. Obtenir l'ancienne Q-value
        old_value = self.q_table[state_idx + (action,)]

        # 2. Calculer le Q-value maximum de l'état suivant
        if done:
            next_max = 0
        else:
            next_max = np.max(self.q_table[next_state_idx])
        
        # 3. Calcul de la nouvelle Q-value (cible)
        new_value = reward + self.gamma * next_max
        
        # 4. Mise à jour de la table Q
        self.q_table[state_idx + (action,)] = old_value + self.alpha * (new_value - old_value)

    def update_epsilon(self):
        """
        Décroissance du taux d'exploration.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon) # autre formule possible

    def get_policy(self):
        """
        Retourne la politique optimale apprise (pour l'affichage/l'évaluation).
        """
        policy = np.zeros((self.q_table_shape[0], self.q_table_shape[1], self.q_table_shape[2], self.q_table_shape[3]), dtype=int)
        
        # On ne peut pas facilement générer une politique pour chaque état possible,
        # mais la Q-table contient la politique (argmax)
        return np.argmax(self.q_table, axis=-1)
