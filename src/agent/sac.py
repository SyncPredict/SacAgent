import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from .actor import Actor
from .critic import CriticNetworks
from ..utils.dir import find_directory


# def clear_memory():
#     torch.cuda.empty_cache()


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic_networks = CriticNetworks(state_dim, action_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device("cpu")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Обновление Critic
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q_target_next_1, q_target_next_2 = self.critic_networks.target_predict(next_states, next_actions)
            min_q_target_next_1 = q_target_next_1 - self.alpha * next_log_pi
            min_q_target_next_2 = q_target_next_2 - self.alpha * next_log_pi
            q_target_1 = rewards + self.gamma * (1 - dones) * min_q_target_next_1
            q_target_2 = rewards + self.gamma * (1 - dones) * min_q_target_next_2

        current_q1, current_q2 = self.critic_networks.predict(states, actions)

        # Вычисление потерь отдельно для каждого критика
        critic_loss_1 = F.mse_loss(current_q1, q_target_1)
        critic_loss_2 = F.mse_loss(current_q2, q_target_2)

        self.critic_networks.optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_networks.optimizer_1.step()

        # Обновление для второго критика
        self.critic_networks.optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_networks.optimizer_2.step()

        # Обновление Actor
        new_actions, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic_networks.predict(states, new_actions)
        q_new_actions = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update целевых сетей
        self.critic_networks.update_targets(self.tau)

    def save_model(self,new_dir, name):
        # Гарантируем наличие директории


        # Обновляем путь для сохранения, используя имя файла
        save_path = os.path.join(new_dir, f'{name}.pth')

        # Сохраняем состояние модели
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_networks_state_dict': self.critic_networks.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_networks_optimizer_1_state_dict': self.critic_networks.optimizer_1.state_dict(),
            'critic_networks_optimizer_2_state_dict': self.critic_networks.optimizer_2.state_dict()
        }, save_path)

    def load_model(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_networks.load_state_dict(checkpoint['critic_networks_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_networks.optimizer_1.load_state_dict(checkpoint['critic_networks_optimizer_1_state_dict'])
        self.critic_networks.optimizer_2.load_state_dict(checkpoint['critic_networks_optimizer_2_state_dict'])
