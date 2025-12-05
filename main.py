from src.environment import LunarEnv
from src.dqn_agent import DQNAgent
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import evaluate

def main():
    # --- Hyperparameters & Configuration ---
    EPISODES = 50000        # Entraînement long pour converger
    GRID_SIZE = 10          
    MAX_PAYLOAD = 3         
    SENSOR_RANGE = 2        # Vue locale 5x5
    LOG_FREQUENCY = 250     # Fréquence d'affichage des logs
    USE_POMDP = False # Flag for setting the problem to a MDP or POMDP
    
    # --- Environment Initialization ---
    env = LunarEnv(
        rows=GRID_SIZE, 
        cols=GRID_SIZE, 
        n_obstacles=15, 
        n_samples=5, 
        max_payload=MAX_PAYLOAD, 
        slippage_chance=0.2, 
        sensor_range=SENSOR_RANGE,
        use_pomdp=USE_POMDP
    )
    
    # --- Agent Initialization ---
    # Input state dimension: Local Grid flattened + Energy + Payload
    state_size = (2 * SENSOR_RANGE + 1)**2 + 2 if USE_POMDP else GRID_SIZE*GRID_SIZE + 2
    action_size = env.action_space_size
    
    agent = DQNAgent(state_size, action_size)
    
    print(f"Starting DQN training on {EPISODES} episodes...", flush=True)
    print(f"Observation Space: {state_size} | Action Space: {action_size}", flush=True)
    print("-" * 50, flush=True)

    start_time = time.time()
    scores = [] 

    # --- Main Training Loop ---
    for e in range(EPISODES):
        state = env.reset() 
        done = False
        total_reward = 0
        
        while not done:
            # 1. Choose action
            action = agent.choose_action(state)
            
            # 2. Execute action
            next_state, reward, done = env.step(action)
            
            # 3. Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Train Neural Network
            agent.replay()
            
            # 5. Update state
            state = next_state
            total_reward += reward
            
        # Update Target Network periodically
        if e % 10 == 0:
            agent.update_target_model()
            
        scores.append(total_reward)
        
        # --- Logging (Périodique & Temps Réel) ---
        if (e + 1) % LOG_FREQUENCY == 0:
            avg_score = np.mean(scores[-LOG_FREQUENCY:])
            print(f"Episode {e+1}/{EPISODES} | Avg Reward ({LOG_FREQUENCY}): {avg_score:.2f} | Epsilon: {agent.epsilon:.6f}", flush=True)

    end_time = time.time()
    print("-" * 50, flush=True)
    print(f"Training finished in {end_time - start_time:.2f} seconds.", flush=True)
    
    # --- Save the Model ---
    model_filename = "lunar_rover_dqn.pth"
    torch.save(agent.model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}", flush=True)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score par Episode', alpha=0.3)
    
    # Moyenne mobile lissée (fenêtre de 100)
    moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
    plt.plot(range(len(moving_avg)), moving_avg, label='Moyenne Mobile (100 épisodes)', color='red')
    
    plt.title('Performance du Deep Q-Learning (Lunar Rover)')
    plt.xlabel('Épisodes')
    plt.ylabel('Score Total')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png') 
    print("Graphique sauvegardé sous 'learning_curve.png'", flush=True)
    
    # --- Lancement de la démo (Commenté pour le serveur) ---
    # print("\nLancement de la démo automatique...", flush=True)
    # try:
    #     evaluate.run_demo()
    # except Exception as e:
    #     print(f"Impossible de lancer la démo (pas d'écran ?): {e}")

if __name__ == "__main__":
    main()