# AA228 Final Project: Autonomous Lunar Rover with Deep RL

## Autonomous Decision-Making for Lunar Regolith Sample Collection under Partial Observability

**Course:** AA228/CS238 - Decision Making under Uncertainty  
**Stanford University** **Team Members:**
* Loïc Poisson
* Martin Yitao
* Jason Gunn
---

## 1. Project Overview

As future lunar missions transition to sustained surface operations, autonomous rovers must operate efficiently in unknown environments with limited sensing capabilities. This project implements a **Deep Reinforcement Learning (Deep RL)** framework that enables a rover to search a stochastic lunar environment, manage its battery life, and collect high-value regolith samples without a pre-loaded map.

Unlike traditional planners that require full state knowledge, our agent operates under **Partial Observability**, making decisions based solely on local sensor readings and internal status.

## 2. Problem Formulation

We model the problem as a **Partially Observable Markov Decision Process (POMDP)** treated as an MDP by using a history of local observations as the state input for a Neural Network.

### State Space (Input Vector)
The agent does not see the global grid coordinates $(x,y)$. Instead, the input state $S$ is a vector composed of:
* **Local Sensor View:** A flattened $(2R+1) \times (2R+1)$ grid centered on the rover, representing the immediate terrain (obstacles, samples, empty space).
* **Internal State:** Current normalized Energy level and Payload capacity.

### Action Space $A$
* **Discrete Actions:** `{Up, Down, Left, Right, Collect}`.
* **Dynamics:** Movement is stochastic; terrain uncertainty introduces a probability of "slippage" where the rover might move to an unintended adjacent cell or stay in place.

### Reward Function $R(s, a)$
* **Sample Collected:** `+50`
* **Step Cost:** `-1` (Encourages efficiency)
* **Obstacle Collision:** `-20`
* **Empty Drill:** `-5`
* **Battery Depletion:** `-100` (Terminal state)

## 3. Methodology: Deep Q-Learning (DQN)

To handle the combinatorial explosion of the state space caused by the local grid view, we utilize **Deep Q-Learning (DQN)**. Instead of a Q-Table, a Neural Network approximates the Q-value function $Q(s, a; \theta)$.

### Network Architecture
* **Input Layer:** Dimension corresponding to the local view size + scalar features (e.g., $5 \times 5 + 2 = 27$ inputs).
* **Hidden Layers:** Two fully connected layers (128 and 64 neurons) with ReLU activation to capture non-linear terrain patterns.
* **Output Layer:** Linear output layer size of $|A|$ (Q-values for each action).

### Key Algorithms Features
1.  **Experience Replay:** Transitions $(s, a, r, s', done)$ are stored in a replay buffer to break correlations between consecutive samples during training.
2.  **Target Network:** A separate network is used to calculate the target Q-values, updated periodically to stabilize training.
3.  **$\epsilon$-Greedy Exploration:** Balances exploration of the unknown map with exploitation of the learned policy.

## 4. Project Structure

```text
lunar_rover_project/
│
├── src/
│   ├── __init__.py
│   ├── environment.py    # LunarEnv with stochasticity & partial observation
│   ├── dqn_agent.py      # PyTorch implementation of DQN (Agent & QNetwork)
│   └── utils.py          # Visualization tools
│
├── main.py               # Training loop and evaluation
├── requirements.txt      # Dependencies (torch, numpy, etc.)
└── README.md             # Project documentation
```



## 5. Installation and Usage
Prerequisites
Python 3.8+
PyTorch
NumPy
Matplotlib

### Installation
* **Clone the repository:**
Bash
git clone [https://github.com/your-username/lunar_rover_project.git](https://github.com/your-username/lunar_rover_project.git)
cd lunar_rover_project
* **Install dependencies:**

Bash
pip install -r requirements.txt

### Running the Training
To train the DQN agent:
Bash
python main.py
The script will initialize the environment, train the Neural Network over thousands of episodes, and periodically print the average reward and epsilon value.

## 6. Future Improvements
* **Convolutional Neural Networks (CNN):** If the sensor range increases significantly, using CNNs to process the local grid as an image would be more efficient than dense layers.

* **Recurrent Q-Learning (DRQN):** Adding LSTM layers to handle state memory, allowing the rover to "remember" obstacles it passed recently but that are now out of view.

* **Continuous Action Space:** Moving to algorithms like PPO or DDPG for more fluid rover control.

Stanford University - AA228/CS238
