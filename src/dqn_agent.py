import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- 1. LE CERVEAU (Réseau de Neurones) ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # Entrée: Vue locale (aplati) + Energie + Payload
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim) # Sortie: Q-value pour chaque action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. L'AGENT DQN ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Mémoire d'expérience (Replay Buffer)
        
        # Hyperparamètres
        self.gamma = 0.95    # Importance du futur
        self.epsilon = 1.0   # Exploration initiale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Modèles (Principal et Cible pour la stabilité)
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """Copie les poids du modèle principal vers le modèle cible"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stocke l'expérience en mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Epsilon-Greedy Policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Conversion numpy -> tensor pour le réseau
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Apprentissage par Experience Replay (Mini-batch)"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Préparation des batchs
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).unsqueeze(1)

        # 1. Prédire Q(s, a) actuel
        current_q_values = self.model(states).gather(1, actions)

        # 2. Prédire Q(s', a') futur avec le Target Network
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q * (1 - dones))

        # 3. Descente de Gradient
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Réduire l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
