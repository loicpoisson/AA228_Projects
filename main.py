from src.environment import LunarEnv
import random

def main():
    # Initialisation de l'environnement
    env = LunarEnv(rows=10, cols=10)
    
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Début de la mission lunaire...")
    
    while not done:
        # Ici, nous remplacerons par l'agent Q-Learning plus tard
        # Pour l'instant, action aléatoire :
        action = random.choice([0, 1, 2, 3, 4]) 
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Affichage basique
        print(f"Pos: {next_state['position']}, Energie: {next_state['energy']}, Reward: {reward}")

    print(f"Mission terminée. Récompense totale : {total_reward}")

if __name__ == "__main__":
    main()
