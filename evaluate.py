# File: evaluate.py
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
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


def save_trajectory(initial_grid, path, filename="trajectory.png"):
    """
    Generates and saves an image of the rover's path on the initial grid.
    """
    rows, cols = initial_grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define colors: 0=Empty(White), 1=Obstacle(Black), 2=Sample(Blue), 3=Base(Green)
    cmap = ListedColormap(['white', 'black', 'blue', 'green'])
    
    # Display the grid (using the initial state to see where samples were)
    ax.imshow(initial_grid, cmap=cmap, vmin=0, vmax=3)

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Extract X (col) and Y (row) coordinates from the path
    # Note: In imshow, X is the column index, Y is the row index.
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]

    # Plot the trajectory line
    ax.plot(path_cols, path_rows, color='red', linewidth=3, marker='o', markersize=5, label='Path')

    # Mark Start (Square) and End (X)
    ax.scatter(path_cols[0], path_rows[0], c='lime', marker='s', s=150, edgecolors='black', label='Start (Base)', zorder=5)
    ax.scatter(path_cols[-1], path_rows[-1], c='red', marker='X', s=150, edgecolors='black', label='End', zorder=5)

    # Add legend and labels
    # Create custom patches for the legend to represent grid items
    legend_elements = [
        patches.Patch(facecolor='white', edgecolor='gray', label='Empty'),
        patches.Patch(facecolor='black', edgecolor='gray', label='Obstacle'),
        patches.Patch(facecolor='blue', edgecolor='gray', label='Sample'),
        patches.Patch(facecolor='green', edgecolor='gray', label='Base'),
        plt.Line2D([0], [0], color='red', lw=3, label='Rover Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    ax.set_title("Autonomous Rover Trajectory")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[Info] Trajectory image saved as '{filename}'")
    plt.close()


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

    initial_grid = env.grid.copy()  # Save initial grid layout (with samples)
    path = [env.rover_pos]          # Initialize path with start position
    
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

        path.append(env.rover_pos)
        
        # Rendering
        render_console(env)
        print(f"Action: {env.actions[action]} | Reward: {reward}")
        time.sleep(0.8) # Pause to follow visually

    print(f"End of demo. Total Score: {total_reward}")

if __name__ == "__main__":
    run_demo()
