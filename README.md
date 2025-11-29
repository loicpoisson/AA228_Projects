# AA228 Final Project: Autonomous Lunar Rover

## Autonomous Decision-Making for Lunar Regolith Sample Collection under Energy and Environmental Uncertainty

**Course:** AA228/CS238 - Decision Making under Uncertainty  
**Stanford University** **Team Members:**
* Loïc Poisson
* Martin Yitao
* Jason Gunn

---

## 1. Project Overview

As future lunar missions transition to sustained surface operations, autonomous rovers will play a pivotal role in identifying and gathering valuable regolith resources. This project focuses on designing a decision-making framework that allows a rover to search an uncertain lunar environment, manage limited energy, and collect high-value samples without continuous human supervision.

We frame this task as a sequential decision-making problem. We simulate a lunar environment as a Grid World containing obstacles (craters), resources (regolith samples), and a base station. The agent uses **Model-Free Q-Learning** to learn an optimal policy to maximize sample collection while minimizing energy consumption and avoiding hazards.

## 2. Problem Formulation

We model the problem as a Markov Decision Process (MDP) where the agent interacts with a stochastic environment.

### State Space $S$
The state of the rover is defined by a tuple:
* **Position:** $(x, y)$ coordinates on the $M \times N$ grid.
* **Energy:** Current battery level $e \in [0, E_{max}]$.
* **Payload:** Amount of regolith collected $p$.
* **Map Knowledge:** (In partial observability scenarios, this includes the local view of the grid).

### Action Space $A$
The rover can take discrete actions at each time step:
* **Move:** `{Up, Down, Left, Right}`. 
  * *Dynamics:* Movement is stochastic; there is a small probability the rover slips or moves to an unintended adjacent cell due to terrain uncertainty.
* **Collect:** Attempt to drill/scoop regolith at the current location.

### Reward Function $R(s, a)$
The reward structure is designed to encourage efficiency and safety:
* **Sample Collected:** `+50` (High reward for finding value).
* **Step Cost:** `-1` (Time penalty to encourage speed).
* **Obstacle Collision:** `-20` (Penalty for hitting rocks/craters).
* **Empty Drill:** `-5` (Penalty for trying to collect where there is no sample).
* **Battery Depletion:** `-100` (Terminal penalty if energy reaches 0).

## 3. Methodology: Q-Learning

Since the environment map is randomly generated and the transition probabilities (stochastic terrain) are not assumed to be known a priori by the agent, we utilize **Q-Learning**, a model-free Reinforcement Learning algorithm.

The agent maintains a Q-Table $Q(s, a)$ and updates values based on the Bellman equation during exploration:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

Where:
* $\alpha$ (Alpha): Learning rate.
* $\gamma$ (Gamma): Discount factor.
* An $\epsilon$-greedy strategy is implemented to balance exploration (trying random moves) and exploitation (using the best known strategy).

## 4. Project Structure

The codebase is organized as follows:

```text
lunar_rover_project/
│
├── src/
│   ├── __init__.py
│   ├── environment.py    # GridWorld simulation, obstacle generation, and physics
│   ├── agent.py          # Q-Learning agent implementation
│   └── utils.py          # Helper functions for visualization and logging
│
├── main.py               # Entry point to run training and simulation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
