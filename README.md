# AA228 Final Project:
Autonomous Lunar RoverAutonomous Decision-Making for Lunar Regolith Sample Collection under Energy and Environmental UncertaintyCourse: AA228/CS238 - Decision Making under Uncertainty
## Team Members:
* Martin Yitao
* Jason Gunn
* Loïc Poisson

*1. Project OverviewAs future lunar missions transition to sustained surface operations, autonomous rovers will play a pivotal role in identifying and gathering valuable regolith resources. This project focuses on designing a decision-making framework that allows a rover to search an uncertain lunar environment, manage limited energy, and collect high-value samples without continuous human supervision.We frame this task as a sequential decision-making problem. We simulate a lunar environment as a Grid World containing obstacles (craters), resources (regolith samples), and a base station. The agent uses Model-Free Q-Learning to learn an optimal policy to maximize sample collection while minimizing energy consumption and avoiding hazards.2. Problem FormulationWe model the problem as a Markov Decision Process (MDP) with elements of partial observability regarding the map layout.State Space $S$The state of the rover is defined by:Position: $(x, y)$ coordinates on the grid.Energy: Current battery level $e \in [0, E_{max}]$.Payload: Amount of regolith collected $p$.Grid Knowledge: (Implicitly represented in the environment simulation).Action Space $A$The rover can take discrete actions at each time step:Move: {Up, Down, Left, Right}. Movement is stochastic; there is a small probability the rover slips or moves to an unintended adjacent cell due to terrain uncertainty.Collect: Attempt to drill/scoop regolith at the current location.Reward Function $R(s, a)$The reward structure encourages efficiency and safety:Sample Collected: $+50$ (High reward for finding value).Step Cost: $-1$ (Time penalty to encourage speed).Obstacle Collision: $-20$ (Penalty for hitting rocks/craters).Empty Drill: $-5$ (Penalty for trying to collect where there is no sample).Battery Depletion: $-100$ (Terminal penalty if energy reaches 0).Dynamics $T(s' | s, a)$Movement: Deterministic in most cases, but stochastic near difficult terrain.Energy: Each move consumes a fixed amount of energy. Collecting samples consumes significantly more energy.3. Methodology: Q-LearningSince the environment map is randomly generated and the transition probabilities (stochastic terrain) are not fully known a priori, we utilize Q-Learning, a model-free Reinforcement Learning algorithm.The agent maintains a Q-Table $Q(s, a)$ and updates values based on the Bellman equation:$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$Where:$\alpha$: Learning rate.$\gamma$: Discount factor.$\epsilon$-greedy strategy is used for exploration vs. exploitation.4. Project StructurePlaintextlunar_rover_project/
│
├── src/
│   ├── __init__.py
│   ├── environment.py    # GridWorld simulation, obstacles, and physics
│   ├── agent.py          # Q-Learning implementation
│   └── utils.py          # Helper functions for visualization
│
├── main.py               # Entry point to run training and testing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
5. Installation and UsagePrerequisitesPython 3.8+NumPyMatplotlib (for visualization)InstallationClone the repository:Bashgit clone https://github.com/your-username/lunar_rover_project.git
cd lunar_rover_project
Install dependencies:Bashpip install -r requirements.txt
Running the SimulationTo train the agent and visualize the results:Bashpython main.py
Arguments can be modified inside main.py to change grid size, obstacle density, or hyperparameters ($\alpha, \epsilon, \gamma$).6. Future ImprovementsPartial Observability: Implementing a full POMDP solver where the rover maintains a belief state about the map structure using only local sensor data.Deep Q-Learning (DQN): Replacing the tabular Q-learning with a Neural Network to handle larger grid sizes and continuous state spaces.Power Budgeting: Adding complex charging dynamics (solar availability based on time of day).Stanford University - AA228/CS238
