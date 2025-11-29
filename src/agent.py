import numpy as np
import random

class LFAAgent:
    """
    Agent Q-Learning avec Approximation Linéaire de Fonction (LFA)
    pour gérer la visibilité partielle (vecteur de features).
    Q(s, a) = W * phi(s, a)
    """
    def __init__(self, action_space_size, feature_vector_size, alpha=0.001, gamma=0.9, epsilon=1.0):
        self.action_space_size = action_space_size
        self.feature_vector_size = feature_vector_size # (2R+1)^2 + 2 = 27 features
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.99999 # Taux de décroissance très lent (50000+ itérations nécessaires)
        
        # --- Matrice de Poids W (W[i, a] est le poids de la feature i pour l'action a) ---
        # Taille: [Feature Size x Action Size]
        self.W = np.zeros((feature_vector_size, action_space_size))

    def _calculate_q_values(self, phi_s):
        """
        Calcule Q(s, a) pour toutes les actions a, via Q = phi(s)^T * W.
        Input: phi_s (vecteur de features)
        Output: vecteur des Q-values pour toutes les actions.
        """
        # (Feature Size) x (Feature Size x Action Size) = (Action Size)
        # La multiplication matricielle est la méthode la plus rapide.
        return phi_s @ self.W
    
    def choose_action(self, phi_s):
        """
        Stratégie Epsilon-Greedy.
        Input: vecteur de features (phi_s)
        """
        # 1. Exploration (random)
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        
        # 2. Exploitation (best action from W)
        q_values = self._calculate_q_values(phi_s)
        return int(np.argmax(q_values))

    def learn(self, phi_s, action, reward, phi_next_s, done):
        """
        Mise à jour des poids W par la règle de la différence temporelle (TD).
        """
        
        # 1. Calculer Q(s', a') max
        if done:
            next_max_q = 0
        else:
            q_values_next = self._calculate_q_values(phi_next_s)
            next_max_q = np.max(q_values_next)
        
        # 2. Calcul de la Cible (Target)
        target = reward + self.gamma * next_max_q
        
        # 3. Calcul de l'Erreur TD
        q_sa = np.dot(phi_s, self.W[:, action]) # Q(s, a) actuel
        td_error = target - q_sa
        
        # 4. Mise à jour des Poids W (Gradient Ascent sur le LFA)
        # Gradient ~ TD_Error * phi(s)
        # W[:, a] <- W[:, a] + alpha * (TD_Error) * phi(s)
        
        # Mise à jour de la colonne de poids W associée à l'action prise
        self.W[:, action] += self.alpha * td_error * phi_s

    def update_epsilon(self):
        """ Décroissance du taux d'exploration. """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
