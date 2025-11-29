from src.environment import LunarEnv
from src.agent import LFAAgent # Changement ici
import numpy as np
import time

def train_agent(episodes=100000): # Beaucoup plus d'épisodes pour l'approximation!
    
    # 1. Définition des paramètres
    GRID_SIZE = 10
    MAX_PAYLOAD = 3
    SENSOR_RANGE = 2 # 5x5 observation grid

    # Calcul de la taille du vecteur de features
    FEATURE_SIZE = (2 * SENSOR_RANGE + 1)**2 + 2 # (5*5) + Energy + Payload = 27
    
    # Initialisation de l'environnement
    env = LunarEnv(rows=GRID_SIZE, cols=GRID_SIZE, n_obstacles=15, n_samples=5, 
                   max_payload=MAX_PAYLOAD, slippage_chance=0.2, sensor_range=SENSOR_RANGE)
    
    # Initialisation de l'agent
    agent = LFAAgent(
        action_space_size=env.action_space_size,
        feature_vector_size=FEATURE_SIZE,
        alpha=0.001, # Taux d'apprentissage réduit pour l'approximation
        gamma=0.95, # Augmenter gamma pour l'horizon long
        epsilon=1.0 
    )

    print(f"Démarrage de l'entraînement LFA (Matrice W de taille {FEATURE_SIZE}x{env.action_space_size})...")
    
    start_time = time.time()
    rewards_history = []
    
    # 2. Boucle d'entraînement
    for episode in range(episodes):
        phi_s = env.reset() # Renvoie le vecteur de features initial
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(phi_s)
            
            # Action dans l'environnement
            phi_next_s, reward, done = env.step(action)
            
            # Apprentissage par l'agent
            agent.learn(phi_s, action, reward, phi_next_s, done)
            
            phi_s = phi_next_s
            total_reward += reward
        
        # 3. Mise à jour Epsilon et Logging
        agent.update_epsilon()
        rewards_history.append(total_reward)
        
        if (episode + 1) % 5000 == 0:
            avg_reward = np.mean(rewards_history[-5000:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (5k): {avg_reward:.2f} | Epsilon: {agent.epsilon:.6f} | W_norm: {np.linalg.norm(agent.W):.2f}")
            
    end_time = time.time()
    
    print("-" * 60)
    print(f"Entraînement LFA terminé en {end_time - start_time:.2f} secondes.")
    
    # Note: La visualisation des résultats sera l'étape suivante.

def main():
    train_agent(episodes=100000)

if __name__ == "__main__":
    main()
