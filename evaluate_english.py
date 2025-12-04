# File: evaluate.py
import torch
import numpy as np
import time
from src.environment import LunarEnv
from src.dqn_agent import DQNAgent

# Function to render the grid in the console
def render_console(env):
    grid_str = ""
    symbols = {0: '.', 1: '#', 2: 'S', 3: 'B'} # . = Empty, # = Obstacle, S = Sample, B = Base
    
    print("\n" + "="*20)
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            if (r, c) == env.rover_pos:
                row_str += " R " # The Rover
            elif env.grid[r, c] in symbols:
                row_str += f" {symbols[env.grid[r, c]]} "
            else:
                row_str += " ? "
        print(row_str + "|")
    print("="*20)
    print(f"Energy: {env.energy:.1f} | Payload: {env.payload}/{env.max_payload}")

def run_demo():
    # 1. Configuration identical to training
    GRID_SIZE = 10
    SENSOR_RANGE = 2
    state_size = (2 * SENSOR_RANGE + 1)**2 + 2 
    
    # 2. Load the environment and the agent
    env = LunarEnv(rows=GRID_SIZE, cols=GRID_SIZE, n_obstacles=15, n_samples=5, sensor_range=SENSOR_RANGE)
    agent = DQNAgent(state_size, env.action_space_size)
    
    # 3. Load trained weights
    try:
        agent.model.load_state_dict(torch.load("lunar_rover_dqn.pth"))
        agent.model.eval() # Evaluation mode
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: The file 'lunar_rover_dqn.pth' does not exist. Run main.py first.")
        return

    # 4. Launch a mission
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("Starting demonstration...")
    render_console(env)
    time.sleep(1)

    while not done and steps < 50: # Safety limit for the demo
        # Purely greedy action (no epsilon exploration)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()
            
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        
        # Rendering
        render_console(env)
        print(f"Action: {env.actions[action]} | Reward: {reward}")
        time.sleep(0.8) # Pause to follow visually

    print(f"End of demo. Total Score: {total_reward}")

if __name__ == "__main__":
    run_demo()
