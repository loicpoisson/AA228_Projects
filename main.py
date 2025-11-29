from src.environment import LunarEnv
from src.agent import QLearningAgent
import time

def train_agent(episodes=20000):
    
    # 1. Définition des paramètres
    GRID_SIZE = 10
    MAX_PAYLOAD = 3
    
    # Initialisation de l'environnement
    env = LunarEnv(rows=GRID_SIZE, cols=GRID_SIZE, n_obstacles=15, n_samples=5, max_payload=MAX_PAYLOAD)
    
    # Initialisation de l'agent
    agent = QLearningAgent(
        action_space_size=env.action_space_size,
        grid_rows=GRID_SIZE, 
        grid_cols=GRID_SIZE, 
        max_energy=env.max_energy, 
        max_payload=MAX_PAYLOAD,
        alpha=0.1, 
        gamma=0.9, 
        epsilon=1.0 # Commence avec 100% d'exploration
    )

    print(f"Démarrage de l'entraînement (Table Q de taille {agent.q_table.shape})...")
    
    start_time = time.time()
    rewards_history = []
    
    # 2. Boucle d'entraînement
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            
            # Action dans l'environnement
            next_state, reward, done = env.step(action)
            
            # Apprentissage par l'agent
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # 3. Mise à jour Epsilon et Logging
        agent.update_epsilon()
        rewards_history.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (1k): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
            
    end_time = time.time()
    
    # 4. Affichage des résultats finaux
    print("-" * 40)
    print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")
    print(f"Meilleur score: {np.max(rewards_history):.2f}")
    
    # Optionnel: Retourner l'agent et l'historique pour la visualisation
    return agent, rewards_history

def main():
    agent, history = train_agent(episodes=50000) # Entraîner sur plus d'épisodes

if __name__ == "__main__":
    main()
