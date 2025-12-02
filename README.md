# AA228 Final Project: Autonomous Lunar Rover with Deep RL

## Autonomous Decision-Making for Lunar Regolith Sample Collection under Partial Observability

**Course:** AA228/CS238 - Decision Making under Uncertainty  
**Stanford University** **Team Members:**
* Loïc Poisson
* Martin Zhou
* Jason Gunn
---

## 1. Project Overview

As future lunar missions transition to sustained surface operations, autonomous rovers must operate efficiently in unknown environments with limited sensing capabilities. This project implements a **Deep Reinforcement Learning (Deep RL)** framework that enables a rover to search a stochastic lunar environment, manage its battery life, and collect high-value regolith samples without a pre-loaded map.

Unlike traditional planners that require full state knowledge, our agent operates under **Partial Observability**, making decisions based solely on local sensor readings and internal status.

## 2. Current Progress & Results (Status Update)

### ✅ What Has Been Achieved
* **Environment Simulation:** Successfully implemented a stochastic GridWorld (`LunarEnv`) with random obstacles, samples, and a base station.
* **Partial Observability:** The agent now navigates using only a local sensor view (radius $R=2$) and internal state (energy, payload), solving the "incomplete knowledge" challenge.
* **Deep Q-Network (DQN):** A fully functional DQN agent is implemented using PyTorch, featuring:
    * **Experience Replay Buffer** for stable learning.
    * **Target Network** updates to prevent divergence.
    * **Epsilon-Greedy** exploration strategy with exponential decay.
* **HPC Deployment:** The training pipeline is optimized and successfully deployed on the Stanford FarmShare cluster (Slurm), enabling long-duration training (50,000+ episodes).

### 📊 Preliminary Results
* **Stability:** The network has demonstrated excellent stability (no divergence/NaNs) during training runs of up to 20,000 episodes.
* **Survival Learning:** The agent has successfully learned to avoid immediate penalties (collisions, time-wasting), improving the average reward from approx. `-700` (random/suicidal) to `-200`.
* **Current Challenge:** The policy is currently stuck in a "survival" local optimum. It avoids death effectively but has not yet fully mastered the long-term planning required to consistently find and return samples for positive rewards.

### 🚧 Work in Progress
* **Hyperparameter Tuning:** We are actively optimizing the **Learning Rate ($\alpha$)** and **Discount Factor ($\gamma$)** to encourage more aggressive exploration and long-term planning. Specifically, increasing $\alpha$ to `0.005` and adding a Learning Rate Scheduler.
* **Exploration Strategy:** Tuning the `epsilon_decay` to ensure the agent explores the map sufficiently before converging to a safe but suboptimal policy.

### 🔜 Next Steps
1.  **Positive Reward Convergence:** Achieve a consistently positive average score by fine-tuning the reward propagation.
2.  **Visualization:** Generate a video/GIF demonstration of the learned policy using the `evaluate.py` script.
3.  **Final Analysis:** Comparative analysis of performance under different sensor ranges and stochasticity levels.

---

## 3. Problem Formulation

We model the problem as a **Partially Observable Markov Decision Process (POMDP)** treated as an MDP by using a history of local observations as the state input for a Neural Network.

### State Space (Input Vector)
The agent does not see the global grid coordinates $(x,y)$. Instead, the input state $S$ is a vector composed of:
* **Local Sensor View:** A flattened $(2R+1) \times (2R+1)$ grid centered on the rover (default $5 \times 5$), representing the immediate terrain.
* **Internal State:** Current normalized Energy level and Payload capacity.

### Action Space $A$
* **Discrete Actions:** `{Up, Down, Left, Right, Collect}`.
* **Dynamics:** Movement is stochastic; terrain uncertainty introduces a probability of "slippage" (20%) where the rover might move to an unintended adjacent cell.

### Reward Function $R(s, a)$
* **Sample Collected:** `+50`
* **Step Cost:** `-1` (Encourages efficiency)
* **Obstacle Collision:** `-20`
* **Empty Drill:** `-5`
* **Battery Depletion:** `-100` (Terminal state)

---

## 4. Methodology: Deep Q-Learning (DQN)

To handle the combinatorial explosion of the state space caused by the local grid view, we utilize **Deep Q-Learning (DQN)**.

### Network Architecture
* **Input Layer:** Dimension corresponding to the local view size + scalar features (e.g., $25 + 2 = 27$ inputs).
* **Hidden Layers:** Two fully connected layers (128 and 64 neurons) with ReLU activation.
* **Output Layer:** Linear output layer size of $|A|$ (Q-values for each action).

### Key Algorithms Features
1.  **Experience Replay:** Breaks correlations between consecutive samples.
2.  **Target Network:** Stabilizes Q-value targets.
3.  **Adaptive Learning Rate:** Uses an `ExponentialLR` scheduler to refine weights as training progresses.

---

## 5. Project Structure

```text
lunar_rover_project/
│
├── src/
│   ├── __init__.py
│   ├── environment.py    # LunarEnv with stochasticity & partial observation
│   ├── dqn_agent.py      # PyTorch implementation of DQN with LR Scheduling
│   └── utils.py          # Helper tools
│
├── main.py               # Main training loop with logging and plotting
├── evaluate.py           # Script to visualize and test the trained agent
├── run_training.sbatch   # Slurm script for cluster training
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```



## 6. Installation and Usage
### Prerequisites
Python 3.8+
PyTorch, NumPy, Matplotlib

### Installation
* **Clone the repository:**
git clone [https://github.com/loicpoisson/AA228_Projects](https://github.com/loicpoisson/AA228_Projects)
cd lunar_rover_project
pip install -r requirements.txt

### Running the Training
To train the DQN agent:
python main.py
The script will initialize the environment, train the Neural Network over 50,000 of episodes, and periodically print the average reward and epsilon value.

### Running the Demo
To watch the trained agent in action (console visualization):
python evaluate.py

## 7. Future Improvements
* **Convolutional Neural Networks (CNN):** If the sensor range increases significantly, using CNNs to process the local grid as an image would be more efficient than dense layers.

* **Recurrent Q-Learning (DRQN):** Adding LSTM layers to handle state memory, allowing the rover to "remember" obstacles it passed recently but that are now out of view.

* **Continuous Action Space:** Moving to algorithms like PPO or DDPG for more fluid rover control.

Stanford University - AA228/CS238
