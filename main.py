from src.environment import LunarEnv
from src.dqn_agent import DQNAgent
import numpy as np
import torch
import time
import evaluate

def main():
    # --- Hyperparameters & Configuration ---
    EPISODES = 1000        # Total number of training episodes
    GRID_SIZE = 10         # Size of the lunar grid (10x10)
    MAX_PAYLOAD = 3        # Maximum samples the rover can carry
    SENSOR_RANGE = 2       # Visibility radius (2 means a 5x5 local grid)
    
    # --- Environment Initialization ---
    env = LunarEnv(
        rows=GRID_SIZE, 
        cols=GRID_SIZE, 
        n_obstacles=15, 
        n_samples=5, 
        max_payload=MAX_PAYLOAD, 
        slippage_chance=0.2, # Probability of stochastic movement (slipping)
        sensor_range=SENSOR_RANGE
    )
    
    # --- Agent Initialization ---
    # Calculate input state dimension: 
    # Local Grid flattened ((2*R+1)^2) + Energy (1) + Payload (1)
    state_size = (2 * SENSOR_RANGE + 1)**2 + 2 
    action_size = env.action_space_size
    
    agent = DQNAgent(state_size, action_size)
    
    print(f"Starting DQN training on {EPISODES} episodes...")
    print(f"Observation Space Size: {state_size}")
    print(f"Action Space Size: {action_size}")
    print("-" * 50)

    start_time = time.time()
    scores = [] # To track rewards over time

    # --- Main Training Loop ---
    for e in range(EPISODES):
        state = env.reset() # Reset env and get initial state
        done = False
        total_reward = 0
        
        while not done:
            # 1. Choose action (Epsilon-Greedy)
            action = agent.choose_action(state)
            
            # 2. Execute action in environment
            next_state, reward, done = env.step(action)
            
            # 3. Store experience in Replay Buffer
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Train the Neural Network (Experience Replay)
            agent.replay()
            
            # 5. Update state
            state = next_state
            total_reward += reward
            
        # Update Target Network periodically to stabilize training
        if e % 10 == 0:
            agent.update_target_model()
            
        # Logging
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:]) # Rolling average of last 100 episodes
        
        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward:.1f} | Avg Score (100): {avg_score:.1f} | Epsilon: {agent.epsilon:.2f}")

    end_time = time.time()
    print("-" * 50)
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # --- Save the Model ---
    model_filename = "lunar_rover_dqn.pth"
    torch.save(agent.model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

    
    # --- VISUALISATION DES RÉSULTATS ---
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score par Episode', alpha=0.3)
    
    # Calcul de la moyenne mobile pour lisser la courbe
    moving_avg = np.convolve(scores, np.ones(50)/50, mode='valid')
    plt.plot(range(len(moving_avg)), moving_avg, label='Moyenne Mobile (50 épisodes)', color='red')
    
    plt.title('Performance du Deep Q-Learning (Lunar Rover)')
    plt.xlabel('Épisodes')
    plt.ylabel('Score Total')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png') # Sauvegarde l'image
    print("Graphique sauvegardé sous 'learning_curve.png'")
    # plt.show()
    
    # --- LANCEMENT AUTOMATIQUE DE LA DÉMO ---
    print("\n" + "="*50)
    print("LANCEMENT DE LA DÉMONSTRATION AUTOMATIQUE")
    print("="*50 + "\n")
    
    # On appelle la fonction run_demo() qui est dans evaluate.py
    evaluate.run_demo()



if __name__ == "__main__":
    main()
