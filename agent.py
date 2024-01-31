import torch
import numpy as np
from neural_network import MarioNet  #AgentNN

from memory_storage import ReplayMemory 

class MarioAgent:
    def __init__(self, input_shape, actions, learning_rate=0.00025, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99999975, min_exploration=0.1, memory_size=100000, batch_size=32, update_frequency=10000):
        self.action_count = actions
        self.step_count = 0

        # Learning parameters
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        # Neural networks
        self.policy_network = MarioNet(input_shape, actions)
        self.target_network = MarioNet(input_shape, actions, frozen=True)

        # Training tools
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

        # Experience replay
        self.memory = ReplayMemory(capacity=memory_size)

    def select_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_count)
        
        observation_tensor = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.policy_network.device)
        return self.policy_network(observation_tensor).argmax().item()
    
    def decrease_exploration(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def synchronize_networks(self):
        if self.step_count % self.update_frequency == 0 and self.step_count > 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, directory, filename):
        torch.save(self.policy_network.state_dict(), os.path.join(directory, filename))

    def load(self, directory, folder, filename):
        file_path = os.path.join(directory, folder, filename)
        self.policy_network.load_state_dict(torch.load(file_path))
        self.target_network.load_state_dict(torch.load(file_path))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.synchronize_networks()
        
        self.optimizer.zero_grad()

        batch = self.memory.sample(self.batch_size).to(self.policy_network.device)

        states, actions, rewards, next_states, dones = batch.unpack()

        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_states).max(dim=1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())

        loss = self.loss_function(current_q_values, expected_q_values)
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.decrease_exploration()
